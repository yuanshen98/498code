#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>
#include "unroll.cuh"

int main()
{

	const int C = 3;
	const int H = 3;
	const int W = 3;
	const int M = 2;
	const int K = 2;

	float X[C*H*W] = { 1,2,0,
		1,1,3,
		0,2,2,
		0,2,1,
		0,3,2,
		1,1,0,
		1,2,1,
		0,1,3,
		3,3,2 };

	float X_out[(H - K + 1)*(W - K + 1)*K*K*C] = { 0 };
	float X_out_cpu[(H - K + 1)*(W - K + 1)*K*K*C] = { 0 };

	auto startGpu = std::chrono::high_resolution_clock::now();
	unrollWithCuda(&X[0], &X_out[0], C, H, W, K);
	auto stopGpu = std::chrono::high_resolution_clock::now();
	unrollWithCpu(&X[0], &X_out_cpu[0], C, H, W, K);
	auto stopCpu = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> gpuElapsed = stopGpu - startGpu;
	std::chrono::duration<double> cpuElapsed = stopCpu - stopGpu;

	std::cout << "GPU time: " << gpuElapsed.count() << std::endl;
	std::cout << "CPU time: " << cpuElapsed.count() << std::endl;

	/*
	for (int i = 0; i <  K*K*C; i++) {
		for (int j = 0; j <(H - K + 1)*(W - K + 1); j++) {
			printf("%f ", X_out[i*(H - K + 1)*(W - K + 1) + j]);
		}
		printf("\n");
	}
	printf("\n");
	for (int i = 0; i < K*K*C; i++) {
		for (int j = 0; j <(H - K + 1)*(W - K + 1); j++) {
			printf("%f ", X_out_cpu[i*(H - K + 1)*(W - K + 1) + j]);
		}
		printf("\n");
	}
	*/

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}