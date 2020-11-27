#pragma once

#include "cuda_runtime.h"


#include "camera.h"
#include "light.h"

#include "common/color.h"
#include "common/vec3.h"

#include "shapes/sphere.h"
#include "shapes/rectangle.h"

#include <vector>

class Scene
{
public:

	Color bg_color = { 0, 0, 0, 255 };

	std::vector<Sphere> spheres;
	std::vector<Rectangle> rectangles;
	std::vector<Light> lights;

	Camera* camera;

	Scene(int frame_width, int frame_height, int numSpheres, int numRectangles, int numLights, Color _bg_color);
	~Scene();

	void AddLight(Vec3 origin, float intensity);
	void AddSphere(Color color, bool reflective, Vec3 origin, float radius);
	void AddRectangle(Color color, bool reflective, Vec3 a, Vec3 b, Vec3 c, bool flipped = false);
	void LoadEntities();

	void moveForward();
	void moveBackward();
	void moveLeft();
	void moveRight();
	void moveUp();
	void moveDown();

	void rotateCamera(double xpos_delta, double ypos_delta);
	void SmootheRotateRight();

	void Render(uchar4* des);
};

