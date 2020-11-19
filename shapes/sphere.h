#pragma once

#include "cuda_runtime.h"

#include "common/color.h"
#include "common/vec3.h"

struct Sphere
{
	Color color;
	double radius;
    Vec3 origin;

	__device__ Sphere(Color _color, Vec3 _origin, double _radius) : color(_color), origin(_origin), radius(_radius) {}
};

