#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "camera.h"
#include "light.h"

#include "common/color.h"
#include "common/ray.h"
#include "common/vec3.h"

#include "shapes/sphere.h"
#include "shapes/rectangle.h"


void RayTrace(
    Camera* camera,
    Color bg_color,
    Sphere* spheres,
    int spheres_cnt,
    Rectangle* rectangles,
    int rectangles_cnt,
    Light* lights,
    int lights_cnt,
    uchar4* des);