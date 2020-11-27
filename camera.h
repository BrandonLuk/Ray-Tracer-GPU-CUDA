#pragma once

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "common/vec3.h"

constexpr double CAMERA_SENSITIVITY = 0.9;

struct Camera {
    Vec3 origin = { 0.0, 0.0, 0.0 };

    double offset = 5.0;

    double pitch = 0.0;
    double yaw = 90.0;
    Vec3 normal = { 0.0, 0.0 , 1.0 };

    Vec3 roll_component = { 0.0, 1.0, 0.0 };

    int frame_width = 1920;
    int frame_height = 1080;

    double FOV = 90.0 * M_PI / 180.0;

    float speed = 5.0f;

    Camera(int _width, int _height) : frame_width{ _width }, frame_height{ _height } {}
};