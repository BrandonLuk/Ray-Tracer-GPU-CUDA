#pragma once

#include "cuda_runtime.h"

#include "vec3.h"

struct Ray
{
    Vec3 origin;
    Vec3 direction;

    __device__ __host__ Ray() {}
    __device__ __host__ Ray(Vec3 o, Vec3 d) : origin(o), direction(d) {}
};
