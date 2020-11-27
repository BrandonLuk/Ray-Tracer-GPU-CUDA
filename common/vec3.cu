#include "vec3.h"

#include <cmath>

__device__ __host__ Vec3::Vec3() : x(0.0), y(0.0), z(0.0) {}

__device__ __host__ Vec3::Vec3(float _x, float _y, float _z) : x{ _x }, y{ _y }, z{ _z } {}


__device__ __host__ Vec3& Vec3::operator+=(const Vec3& rhs)
{
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;

    return *this;
}

__device__ __host__ Vec3& Vec3::operator-=(const Vec3& rhs)
{
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;

    return *this;
}

__device__ __host__ float Vec3::Magnitude() const
{
#ifdef __CUDA_ARCH__
    return norm3df(x, y, z);
#else
    return sqrt(x * x + y * y + z * z);
#endif
}

__device__ __host__ Vec3 Vec3::Normalize()
{
    Vec3 v;
    float magnitude = Magnitude();

#ifdef __CUDA_ARCH__
    v.x = __fdividef(x, magnitude);
    v.y = __fdividef(y, magnitude);
    v.z = __fdividef(z, magnitude);
#else
    v.x = x / magnitude;
    v.y = y / magnitude;
    v.z = z / magnitude;
#endif
    return v;
}

__device__ __host__ Vec3 Vec3::CrossProduct(const Vec3& v)
{
    Vec3 result;

#ifdef __CUDA_ARCH__
    result.x = __fmaf_rn(y, v.z, -z * v.y);
    result.y = __fmaf_rn(z, v.x, -x * v.z);
    result.z = __fmaf_rn(x, v.y, -y * v.x);
#else
    result.x = (y * v.z) - (z * v.y);
    result.y = (z * v.x) - (x * v.z);
    result.z = (x * v.y) - (y * v.x);
#endif
    return result;
}

__device__ __host__ float Vec3::DotProduct(const Vec3& v) const
{
#ifdef __CUDA_ARCH__
    return __fmaf_rn(x, v.x, __fmaf_rn(y, v.y, z * v.z));
#else
    return (x * v.x) + (y * v.y) + (z * v.z);
#endif
}

__device__ __host__ float Vec3::Distance(const Vec3& v)
{

#ifdef __CUDA_ARCH__
    return norm3df(v.x - x, v.y - y, v.z - z);
#else
    return sqrt((v.x - x) * (v.x - x) +
        (v.y - y) * (v.y - y) +
        (v.z - z) * (v.z - z));
#endif
}

__device__ __host__ Vec3 operator-(const Vec3& vec)
{
    return Vec3(-vec.x, -vec.y, -vec.z);
}

__device__ __host__ Vec3 operator+(const Vec3& lhs, const Vec3& rhs)
{
    return Vec3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

__device__ __host__ Vec3 operator-(const Vec3& lhs, const Vec3& rhs)
{
    return Vec3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

__device__ __host__ Vec3 operator*(float d, const Vec3& v)
{
    return Vec3(d * v.x, d * v.y, d * v.z);
}

__device__ __host__ Vec3 operator*(const Vec3& v, float d)
{
    return d * v;
}