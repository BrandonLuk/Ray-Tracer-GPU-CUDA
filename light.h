#pragma once

#include "common/vec3.h"

struct Light
{
	Vec3 origin = { 0.0 , 0.0, 0.0 };
	float intensity = 1.0;
};