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
	auto safe_add = [](const unsigned char& add1, const unsigned char& add2) ->unsigned char {
		int res = add1 + add2;
		if (res < 0)
			return 0;
		else if (res > COLOR_VALUE_MAX)
			return static_cast<unsigned char>(COLOR_VALUE_MAX);
		else
			return static_cast<unsigned char>(res);
	};

	return { safe_add(data.x, other.data.x),
			 safe_add(data.y, other.data.y),
			 safe_add(data.z, other.data.z),
			 safe_add(data.w, other.data.w) };
}

__device__ __host__ Color operator*(const double& d, const Color& c)
{
	auto safe_mul = [&d](const unsigned char& additive) -> unsigned char {
		double res = additive * d;
		if (res < 0)
			return 0;
		else if (res > COLOR_VALUE_MAX)
			return static_cast<unsigned char>(COLOR_VALUE_MAX);
		else
			return static_cast<unsigned char>(res);
	};

	return { safe_mul(c.data.x),
			 safe_mul(c.data.y),
			 safe_mul(c.data.z),
			 safe_mul(c.data.w) };
}
__device__ __host__ Color operator*(const Color& c, const double& d)
{
	return d * c;
}