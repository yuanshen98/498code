#ifndef COMPARE_CUH
#define COMPARE_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <math.h>

#include <stdio.h>

void forwardWithMatmul(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K);
void forwardWithLoop(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K);

#endif
