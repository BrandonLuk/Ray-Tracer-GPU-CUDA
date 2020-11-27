#pragma once

#include "cuda_runtime.h"


struct Vec3
{
    float x, y, z;

    __device__ __host__ Vec3();
    __device__ __host__ Vec3(float _x, float _y, float _z);

    __device__ __host__ Vec3& operator+=(const Vec3& rhs);
    __device__ __host__ Vec3& operator-=(const Vec3& rhs);


    __device__ __host__ float Magnitude() const;
    __device__ __host__ Vec3 Normalize();
    __device__ __host__ Vec3 CrossProduct(const Vec3& v);
    __device__ __host__ float DotProduct(const Vec3& v) const;
    __device__ __host__ float Distance(const Vec3& v);
};

__device__ __host__ Vec3 operator-(const Vec3& vec);
__device__ __host__ Vec3 operator+(const Vec3& lhs, const Vec3& rhs);
__device__ __host__ Vec3 operator-(const Vec3& lhs, const Vec3& rhs);
__device__ __host__ Vec3 operator*(float d, const Vec3& v);
__device__ __host__ Vec3 operator*(const Vec3& v, float d);