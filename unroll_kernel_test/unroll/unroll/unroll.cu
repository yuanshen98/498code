#include "unroll.cuh"

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void unrollKernel(int C, int H, int W, int K, float* X, float* X_out)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int W_unroll = H_out*W_out;
	int c, s, h_out, w_out, h_unroll, w_unroll, w_base, k1, k2;

	if (tid < C*W_unroll) {
		c = tid / W_unroll;
		s = tid % W_unroll;
		h_out = s / W_out;
		w_out = s % W_out;
		h_unroll = h_out*W_out + w_out;
		w_base = c*K*K;
		for (k1 = 0; k1 < K; k1++) {
			for (k2 = 0; k2 < K; k2++) {
				w_unroll = w_base + k1*K + k2;
				X_out[w_unroll*W_unroll+h_unroll] = X[c*H*W + (h_out + k1)*W + w_out + k2];
			}
		}
	}

}

void unrollWithCuda(float* X, float* X_out, int C, int H, int W, int K) {
	float* dev_X;
	float* dev_X_out;

	cudaSetDevice(0);
	cudaMalloc((void**)&dev_X, (size_t)(H*W*C * sizeof(float)));
	cudaMalloc((void**)&dev_X_out, (size_t)((H-K+1)*(W-K+1)*K*K*C*sizeof(float)));

	cudaMemcpy(dev_X, X, (size_t)(H*W*C * sizeof(float)), cudaMemcpyHostToDevice);

	dim3 blockDim(H*W*C, 1, 1);
	dim3 gridDim(1, 1, 1);
	unrollKernel << <gridDim, blockDim >> > (C, H, W, K, dev_X, dev_X_out);

	cudaMemcpy(X_out, dev_X_out, (size_t)((H - K + 1)*(W - K + 1)*K*K*C * sizeof(float)), cudaMemcpyDeviceToHost);

	cudaFree(dev_X);
	cudaFree(dev_X_out);
}

void unrollWithCpu(float* X, float* X_out, int C, int H, int W, int K) {
	int w_start, w_unroll, h_unroll;
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int W_unroll = H_out*W_out;
	for (int c = 0; c < C; c++) {
		w_start = c*K*K;
		for (int k1 = 0; k1 < K; k1++) {
			for (int k2 = 0; k2 < K; k2++) {
				for (int h = 0; h < H_out; h++) {
					for (int w = 0; w < W_out; w++) {
						w_unroll = w_start + k1*K + k2;
						h_unroll = h*W_out + w;
						X_out[w_unroll*W_unroll + h_unroll] = X[c*H*W + (h + k1)*W + w + k2];
					}
				}
			}
		}
	}
}