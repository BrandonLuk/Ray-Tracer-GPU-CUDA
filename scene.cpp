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

	bg_color = _bg_color;
}


Scene::~Scene()
{
	cudaErrorChk(cudaFree(camera));
}

void Scene::AddLight(Vec3 origin, float intensity)
{
	lights.push_back(Light{ origin, intensity });
}

void Scene::AddSphere(Color color, bool reflective, Vec3 origin, float radius)
{
	spheres.push_back(Sphere(color, reflective, origin, radius));
}

void Scene::AddRectangle(Color color, bool reflective, Vec3 a, Vec3 b, Vec3 c, bool flipped)
{
	rectangles.push_back(Rectangle(color, reflective, a, b, c, flipped));
}

void Scene::LoadEntities()
{
	CpyEntitiesToDevice(spheres.data(), spheres.size(), rectangles.data(), rectangles.size(), lights.data(), lights.size());
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

void Scene::moveUp()
{
	camera->origin += camera->roll_component * camera->speed;
}

void Scene::moveDown()
{
	camera->origin -= camera->roll_component * camera->speed;
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

void Scene::SmootheRotateRight()
{
	camera->yaw -= CAMERA_SENSITIVITY;

	double yaw_rads = camera->yaw * M_PI / 180.0;
	double pitch_rads = camera->pitch * M_PI / 180.0;

	camera->normal.x = cos(yaw_rads) * cos(pitch_rads);
	camera->normal.y = sin(pitch_rads);
	camera->normal.z = sin(yaw_rads) * cos(pitch_rads);
	camera->normal = camera->normal.Normalize();
}

void Scene::Render(uchar4* des)
{
	RayTrace(camera, bg_color, des);
}