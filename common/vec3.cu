#include "vec3.h"

#include <cmath>

__device__ __host__ Vec3::Vec3() : x(0.0), y(0.0), z(0.0) {}

__device__ __host__ Vec3::Vec3(double _x, double _y, double _z) : x{ _x }, y{ _y }, z{ _z } {}


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

__device__ __host__ double Vec3::Magnitude() const
{
    return sqrt(x * x + y * y + z * z);
}

__device__ __host__ Vec3 Vec3::Normalize()
{
    Vec3 v;
    double magnitude = Magnitude();

    v.x = x / magnitude;
    v.y = y / magnitude;
    v.z = z / magnitude;

    return v;
}

__device__ __host__ Vec3 Vec3::CrossProduct(const Vec3& v)
{
    Vec3 result;

    result.x = (y * v.z) - (z * v.y);
    result.y = (z * v.x) - (x * v.z);
    result.z = (x * v.y) - (y * v.x);

    return result;
}

__device__ __host__ double Vec3::DotProduct(const Vec3& v) const
{
    return (x * v.x) + (y * v.y) + (z * v.z);
}

__device__ __host__ double Vec3::Distance(const Vec3& v)
{
    return sqrt((v.x - x) * (v.x - x) +
        (v.y - y) * (v.y - y) +
        (v.z - z) * (v.z - z));
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

__device__ __host__ Vec3 operator*(double d, const Vec3& v)
{
    Vec3 result;

    result.x = d * v.x;
    result.y = d * v.y;
    result.z = d * v.z;

    return result;
}

__device__ __host__ Vec3 operator*(const Vec3& v, double d)
{
    return d * v;
}