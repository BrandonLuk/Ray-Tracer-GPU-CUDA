#pragma once

#include "cuda_runtime.h"


struct Vec3
{
    double x, y, z;

    __device__ __host__ Vec3();
    __device__ __host__ Vec3(double _x, double _y, double _z);

    __device__ __host__ Vec3& operator+=(const Vec3& rhs);
    __device__ __host__ Vec3& operator-=(const Vec3& rhs);


    __device__ __host__ double Magnitude() const;
    __device__ __host__ Vec3 Normalize();
    __device__ __host__ Vec3 CrossProduct(const Vec3& v);
    __device__ __host__ double DotProduct(const Vec3& v) const;
    __device__ __host__ double Distance(const Vec3& v);
};

__device__ __host__ Vec3 operator-(const Vec3& vec);
__device__ __host__ Vec3 operator+(const Vec3& lhs, const Vec3& rhs);
__device__ __host__ Vec3 operator-(const Vec3& lhs, const Vec3& rhs);
__device__ __host__ Vec3 operator*(double d, const Vec3& v);
__device__ __host__ Vec3 operator*(const Vec3& v, double d);