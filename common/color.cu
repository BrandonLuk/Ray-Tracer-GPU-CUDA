#include "color.h"

#include "cuda_runtime.h"

__device__ __host__ Color::Color()
{
	data.x = 0;
	data.y = 0;
	data.z = 0;
	data.w = 0;
}

__device__ __host__ Color::Color(unsigned char _r, unsigned char _g, unsigned char _b, unsigned char _a) {
	data.x = _r;
	data.y = _g;
	data.z = _b;
	data.w = _a;
}

__device__ __host__ Color::operator uchar4() const
{
	return data;
}

__device__ __host__ Color Color::operator+(const Color& other)
{
	return { unsigned char(1U * data.x + other.data.x),
			unsigned char(1U * data.y + other.data.y),
			unsigned char(1U * data.z + other.data.z),
			data.w };
}

__device__ __host__ Color operator*(const float& d, const Color& c)
{
	return { ((unsigned char)(1U * c.data.x * d)),
			((unsigned char)(1U * c.data.y * d)),
			((unsigned char)(1U * c.data.z * d)),
			c.data.w };
}
__device__ __host__ Color operator*(const Color& c, const float& d)
{
	return d * c;
}