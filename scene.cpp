#include "scene.h"
#include "light.h"
#include "kernel.cuh"

#include "common/cuda_error_check.h"

#include "shapes/rectangle.h"
#include "shapes/sphere.h"

#include <cmath>

Scene::Scene(int frame_width, int frame_height, int numSpheres, int numRectangles, int num_lights, Color _bg_color)
{
	cudaErrorChk(cudaMallocManaged(&camera, sizeof(Camera)));
	*camera = Camera(frame_width, frame_height);

	cudaErrorChk(cudaMallocManaged(&spheres, numSpheres * sizeof(Sphere)));
	cudaErrorChk(cudaMallocManaged(&rectangles, numRectangles * sizeof(Rectangle)));
	cudaErrorChk(cudaMallocManaged(&lights,   num_lights * sizeof(Light)));

	spheres_cnt = 0;
	rectangles_cnt = 0;
	lights_cnt = 0;

	bg_color = _bg_color;
}


Scene::~Scene()
{
	cudaErrorChk(cudaFree(spheres));
	cudaErrorChk(cudaFree(rectangles));
	cudaErrorChk(cudaFree(lights));
}

void Scene::AddLight(Vec3 origin, double intensity)
{
	lights[lights_cnt++] = Light{ origin, intensity };
}

void Scene::AddSphere(Color color, Vec3 origin, double radius)
{
	spheres[spheres_cnt] = Sphere(color, origin, radius);
	spheres_cnt += 1;
}

void Scene::AddRectangle(Color color, Vec3 a, Vec3 b, Vec3 c, bool flipped)
{
	rectangles[rectangles_cnt] = Rectangle(color, a, b, c, flipped);
	rectangles_cnt += 1;
}

void Scene::moveForward()
{
	camera->origin += camera->normal * camera->speed;
}

void Scene::moveBackward()
{
	camera->origin -= camera->normal * camera->speed;
}

void Scene::moveLeft()
{
	camera->origin += camera->normal.CrossProduct(camera->roll_component).Normalize() * camera->speed;
}

void Scene::moveRight()
{
	camera->origin -= camera->normal.CrossProduct(camera->roll_component).Normalize() * camera->speed;
}

void Scene::rotateCamera(double xpos_delta, double ypos_delta)
{
	camera->yaw += xpos_delta * CAMERA_SENSITIVITY;
	camera->pitch += ypos_delta * CAMERA_SENSITIVITY;

	if (camera->pitch > 89.0)
		camera->pitch = 89.0;
	else if (camera->pitch < -89.0)
		camera->pitch = -89.0;

	double yaw_rads = camera->yaw * M_PI / 180.0;
	double pitch_rads = camera->pitch * M_PI / 180.0;

	camera->normal.x = cos(yaw_rads) * cos(pitch_rads);
	camera->normal.y = sin(pitch_rads);
	camera->normal.z = sin(yaw_rads) * cos(pitch_rads);
	camera->normal = camera->normal.Normalize();
}

void Scene::Render(uchar4* des)
{
	RayTrace(camera, bg_color, spheres, spheres_cnt, rectangles, rectangles_cnt, lights, lights_cnt, des);
}