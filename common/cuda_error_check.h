#pragma once

#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>

#define cudaErrorChk(func) { cudaFuncAssert((func), __FILE__, __LINE__); }
inline void cudaFuncAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "cudaFuncAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        cudaDeviceReset();
        exit(code);
    }
}