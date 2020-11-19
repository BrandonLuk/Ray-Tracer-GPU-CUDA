#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

#include "camera.h"
#include "kernel.cuh"
#include "scene.h"

#include "common/color.h"
#include "common/cuda_error_check.h"
#include "common/vec3.h"

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

int WINDOW_WIDTH = 1920;
int WINDOW_HEIGHT = 1080;
const std::string WINDOW_TITLE_BASE = "GPU RayTracer - Brandon L.";

bool quit_program = false;

unsigned long long int num_frames = 0;
double time_delta = 0.0;
std::chrono::system_clock::time_point last_time;
std::chrono::duration<double> elapsed;

double cursor_xpos_delta;
double cursor_ypos_delta;
int cursor_horizontal_inversion = -1;
int cursor_vertical_inversion   = -1;

///////////////////////////////////////////////
// OpenGL/Cuda rendering
///////////////////////////////////////////////

GLuint gl_texturePtr;
GLuint gl_pixelBufferObject;
cudaGraphicsResource* cgr;
uchar4* h_textureBufData;
uchar4* d_textureBufData;

///////////////////////////////////////////////
// Key flags
///////////////////////////////////////////////

bool w_key_pressed = false;
bool a_key_pressed = false;
bool s_key_pressed = false;
bool d_key_pressed = false;

///////////////////////////////////////////////
// GLFW callbacks
///////////////////////////////////////////////

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        quit_program = true;

    else if (key == GLFW_KEY_W && action == GLFW_PRESS)
        w_key_pressed = true;
    else if (key == GLFW_KEY_W && action == GLFW_RELEASE)
        w_key_pressed = false;

    else if (key == GLFW_KEY_A && action == GLFW_PRESS)
        a_key_pressed = true;
    else if (key == GLFW_KEY_A && action == GLFW_RELEASE)
        a_key_pressed = false;

    else if (key == GLFW_KEY_S && action == GLFW_PRESS)
        s_key_pressed = true;
    else if (key == GLFW_KEY_S && action == GLFW_RELEASE)
        s_key_pressed = false;

    else if (key == GLFW_KEY_D && action == GLFW_PRESS)
        d_key_pressed = true;
    else if (key == GLFW_KEY_D && action == GLFW_RELEASE)
        d_key_pressed = false;
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    static double last_xpos = WINDOW_WIDTH / 2;
    static double last_ypos = WINDOW_HEIGHT / 2;

    cursor_xpos_delta = xpos - last_xpos;
    cursor_ypos_delta = ypos - last_ypos;

    last_xpos = xpos;
    last_ypos = ypos;
}

void initBuffers(Camera* camera)
{
    h_textureBufData = new uchar4[(size_t)WINDOW_HEIGHT * WINDOW_WIDTH];

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_texturePtr);
    glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_textureBufData);

    glGenBuffers(1, &gl_pixelBufferObject);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, (size_t)WINDOW_HEIGHT * WINDOW_WIDTH * sizeof(uchar4), h_textureBufData, GL_STREAM_COPY);

    cudaGraphicsGLRegisterBuffer(&cgr, gl_pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void moveCamera(Scene& scene)
{
    if (w_key_pressed)
        scene.moveForward();
    if (a_key_pressed)
        scene.moveLeft();
    if (s_key_pressed)
        scene.moveBackward();
    if (d_key_pressed)
        scene.moveRight();
}

void rotateCamera(Scene& scene)
{
    if (cursor_xpos_delta != 0.0 || cursor_ypos_delta != 0.0)
        scene.rotateCamera(cursor_xpos_delta * cursor_horizontal_inversion,
                           cursor_ypos_delta * cursor_vertical_inversion);

    cursor_xpos_delta = 0.0;
    cursor_ypos_delta = 0.0;
}

void UpdateFPS(GLFWwindow* window)
{
    static const unsigned int WINDOW_UPDATE_INTERVAL = 4;
    
    num_frames++;
    elapsed = std::chrono::system_clock::now() - last_time;
    last_time = std::chrono::system_clock::now();
    time_delta += elapsed.count();

    if (time_delta > 1.0 / WINDOW_UPDATE_INTERVAL)
    {
        double FPS = num_frames / time_delta;
        num_frames = 0;
        time_delta -= 1.0 / WINDOW_UPDATE_INTERVAL;

        std::stringstream title;

        title << WINDOW_TITLE_BASE << " FPS: " << FPS;
        glfwSetWindowTitle(window, title.str().c_str());
    }
}

void BuildScene(Scene& scene)
{
    scene.AddRectangle(Color(100, 100, 100, 255), Vec3(-1000, 400, 1000), Vec3(1000, 400, 1000), Vec3(-1000, -400, 1000));          // Front wall
    scene.AddRectangle(Color(100, 50, 50, 255), Vec3(-1000, -400, 1000), Vec3(1000, -400, 1000), Vec3(-1000, -400, -1000));       // Floor
    scene.AddRectangle(Color(0, 0, 100, 255), Vec3(-1000, 400, 1000), Vec3(1000, 400, 1000), Vec3(-1000, 400, -1000), true);    // Ceiling
    scene.AddRectangle(Color(100, 100, 100, 255), Vec3(-1000, 400, 1000), Vec3(-1000, 400, -1000), Vec3(-1000, -400, 1000), true);  // Left wall
    scene.AddRectangle(Color(100, 100, 100, 255), Vec3(1000, 400, 1000), Vec3(1000, 400, -1000), Vec3(1000, -400, 1000));           // Right wall

    scene.AddSphere(Color(100, 0, 0, 255), Vec3(50, 20, 100), 10);
    scene.AddSphere(Color(0, 100, 0, 255), Vec3(50, 70, 100), 20);
    scene.AddSphere(Color(0, 0, 100, 255), Vec3(50, 120, 150), 30);

    scene.AddLight({ -50.0, 50.0, 0.0 }, 1.5);
}

void renderScene(Scene& scene)
{
    cudaErrorChk(cudaGraphicsMapResources(1, &cgr, 0));
    size_t num_bytes;
    cudaErrorChk(cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufData, &num_bytes, cgr));

    scene.Render(d_textureBufData);

    cudaErrorChk(cudaGraphicsUnmapResources(1, &cgr, 0));

    glColor3f(1.0f, 1.0f, 1.0f);
    glBindTexture(GL_TEXTURE_2D, gl_texturePtr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pixelBufferObject);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(float(WINDOW_WIDTH), 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(float(WINDOW_WIDTH), float(WINDOW_HEIGHT));
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0.0f, float(WINDOW_HEIGHT));
    glEnd();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}


int main(void)
{
    /* Initialize the GLFW library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE_BASE.data(), NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    glewInit();

    /* Change cursor mode */
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (glfwRawMouseMotionSupported())
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

    /* Assign callbacks to window */
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    int buf_w, buf_h;
    glfwGetFramebufferSize(window, &buf_w, &buf_h);
    glViewport(0, 0, buf_w, buf_h);
    glOrtho(0.0f, WINDOW_WIDTH, WINDOW_HEIGHT, 0.0f, 0.0f, 1.0f);

    glfwSetCursorPos(window, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);

    Scene scene(WINDOW_WIDTH, WINDOW_HEIGHT, 10, 10, 10, Color(100, 0, 0, 255));
    BuildScene(scene);

    initBuffers(scene.camera);

    last_time = std::chrono::system_clock::now();

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window) && !quit_program)
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        renderScene(scene);        

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
        rotateCamera(scene);
        moveCamera(scene);

        UpdateFPS(window);
    }

    delete(h_textureBufData);
    glfwTerminate();
    return 0;
}