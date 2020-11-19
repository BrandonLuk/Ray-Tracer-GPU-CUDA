#pragma once

#include "cuda_runtime.h"


#include "camera.h"
#include "light.h"

#include "common/color.h"
#include "common/vec3.h"

#include "shapes/sphere.h"
#include "shapes/rectangle.h"

class Scene
{
public:

	Color bg_color = { 0, 0, 0, 255};

	Sphere* spheres;
	int spheres_cnt;
	Rectangle* rectangles;
	int rectangles_cnt;

	Light* lights;
	int lights_cnt = 0;

	Camera* camera;

	Scene(int frame_width, int frame_height, int numSpheres, int numRectangles, int numLights, Color _bg_color);
	~Scene();

	void AddLight(Vec3 origin, double intensity);
	void AddSphere(Color color, Vec3 origin, double radius);
	void AddRectangle(Color color, Vec3 a, Vec3 b, Vec3 c, bool flipped = false);

	void moveForward();
	void moveBackward();
	void moveLeft();
	void moveRight();

	void rotateCamera(double xpos_delta, double ypos_delta);

	void Render(uchar4* des);
};

