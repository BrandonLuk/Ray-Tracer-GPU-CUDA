#pragma once

#include "cuda_runtime.h"

#include "common/color.h"
#include "common/vec3.h"

struct  Rectangle
{
    Color color;
    bool reflective;

    Vec3 normal; // Normal vector of the plane
    Vec3 a; // "Main" point which meets at the right angle made by B and C
    Vec3 b; // Connected to A
    Vec3 c; // Connected to A

    __device__ __host__ Rectangle(Color _color, bool _reflective, Vec3 _a, Vec3 _b, Vec3 _c, bool flipped = false) : color(_color), reflective(_reflective), a(_a), b(_b), c(_c)
    {
        normal = ((b - a).CrossProduct(c - a)).Normalize();

        if (flipped)
            normal = -normal;
    }
};


