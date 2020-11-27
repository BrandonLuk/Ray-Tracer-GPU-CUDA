/*
*	Declares the Color struct which represents colors using the RGBA color model.
*/

#pragma once

#include "cuda_runtime.h"

constexpr int COLOR_DEPTH = 4;
constexpr int COLOR_VALUE_MAX = 255; // Max value of an unsigned char


struct Color
{
	uchar4 data;

	__device__ __host__ Color();
	__device__ __host__ Color(unsigned char _r, unsigned char _g, unsigned char _b, unsigned char _a);

	__device__ __host__ operator uchar4() const;
	__device__ __host__ Color operator+(const Color& other);
};

__device__ __host__ Color operator*(const float& d, const Color& c);
__device__ __host__ Color operator*(const Color& c, const float& d);