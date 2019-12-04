
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


void unrollWithCuda(float* X, float* X_out, int C, int H, int W, int K);
void unrollWithCpu(float* X, float* X_out, int C, int H, int W, int K);