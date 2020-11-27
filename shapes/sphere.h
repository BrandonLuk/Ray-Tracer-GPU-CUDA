#pragma once

#include "cuda_runtime.h"

#include "common/color.h"
#include "common/vec3.h"

struct Sphere
{
	Color color;
	bool reflective;
	Vec3 origin;
	float radius;

	__device__ Sphere(Color _color, bool _reflective, Vec3 _origin, float _radius) : color(_color), reflective(_reflective), radius(_radius), origin(_origin) {}
};