#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>
#include <stdlib.h>
#include "compare.cuh"

int main()
{

	const int C = 4;
	const int H = 10;
	const int W = 10;
	const int M = 2;
	const int K = 5;
	const int B = 6;

	const int H_out = H - K + 1;
	const int W_out = W - K + 1;

	float x[C*H*W*B];
	float k[K*K*M*C];
	float y_1[B*M*H_out*W_out];
	float y_2[B*M*H_out*W_out];

	//generate random test data between 1 and 10 with int truncation
	srand(0);

	for (int i = 0; i < C*H*W*B; i++) {
		x[i] = (float)((int)(rand() % 10));
	}
	for (int i = 0; i < K*K*M*C; i++) {
		k[i] = (float)((int)(rand() % 10));
	}

	auto timeLoopStart = std::chrono::high_resolution_clock::now();
	forwardWithLoop(y_1, x, k, B, M, C, H, W, K);
	auto timeLoopEnd = std::chrono::high_resolution_clock::now();
	forwardWithMatmul(y_2, x, k, B, M, C, H, W, K);
	auto timeMulEnd = std::chrono::high_resolution_clock::now();

	auto loopTime = timeLoopEnd - timeLoopStart;
	auto mulTime = timeMulEnd - timeLoopEnd;
	std::cout << "Loop Time: " << loopTime.count() << std::endl;
	std::cout << "MatMul Time: " << mulTime.count() << std::endl;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	for (int i = 0; i < B*M*H_out*W_out; i++) {
		if (abs(y_1[i] - y_2[i]) > 0.0001) {
			printf("%d %f %f\n", i, y_1[i], y_2[i]);
		}
	}
	printf("done!");

	return 0;
}

