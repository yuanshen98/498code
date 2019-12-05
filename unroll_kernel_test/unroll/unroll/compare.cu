#include "compare.cuh"
__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

	/*
	The goal here is to be correct AND fast.
	We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
	*/


	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	int b = blockDim.x * blockIdx.x + threadIdx.x;
	const int H_out = H - K + 1;
	const int W_out = W - K + 1;

	if (b < B) // for each image in the batch
	{
		for (int m = 0; m < M; m++)         // for each output feature maps
			for (int h = 0; h < H_out; h++) // for each output element
				for (int w = 0; w < W_out; w++)
				{
					y4d(b, m, h, w) = 0;
					for (int c = 0; c < C; c++)     // sum over all input feature maps
						for (int p = 0; p < K; p++) // KxK filter
							for (int q = 0; q < K; q++)
								y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
				}
	}

#undef y4d
#undef x4d
#undef k4d
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
				X_out[w_unroll*W_unroll + h_unroll] = X[c*H*W + (h_out + k1)*W + w_out + k2];
			}
		}
	}

}

#define TILE_WIDTH 32
#define TILE_WIDTH_FLOAT 32.0

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
	int numARows, int numAColumns,
	int numBRows, int numBColumns,
	int numCRows, int numCColumns) {

	__shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

	float multVal = 0;

	//printf("%d %d\n", numAColumns, numAColumns/16);

	//if((row * numCColumns + col) < (numCRows*numCColumns)) {

	//printf("row %d col %d \n", row, col);

		//compute the number of tiles needed 
	for (int i = 0; i < ceilf(numAColumns / TILE_WIDTH_FLOAT); i++) {

		//each thread loads its bit
		if (row * numAColumns + i * TILE_WIDTH + threadIdx.x < numARows*numAColumns) {
			subTileA[threadIdx.y][threadIdx.x] = A[row * numAColumns + i * TILE_WIDTH + threadIdx.x];
			//subTileB[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * numBColumns + col];
			//printf("%d, %d, %d, %d, %f\n", row, i, threadIdx.y, threadIdx.x, A[row * numAColumns + i * TILE_WIDTH + threadIdx.x]);
		}
		if ((i * TILE_WIDTH + threadIdx.y) * numBColumns + col < numBRows*numBColumns) {
			subTileB[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * numBColumns + col];
		}

		__syncthreads();	
		if (row < numCRows && col < numCColumns) {
			for (int j = 0; j < TILE_WIDTH; j++) {
				if (i*TILE_WIDTH + j < numAColumns) {
					//printf("%d, %d, %f, %f\n", threadIdx.y, threadIdx.x, subTileA[threadIdx.y][j], subTileB[j][threadIdx.x]);
					multVal += subTileA[threadIdx.y][j] * subTileB[j][threadIdx.x];
				}
			}
			C[row*numCColumns + col] = multVal;
		}
		__syncthreads();

	}
}


void forwardWithMatmul(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
	float* dev_y;
	float* dev_x;
	float* dev_k;
	float* dev_X_out;

	int H_out = H - K + 1;
	int W_out = W - K + 1;

	cudaMalloc((void**)&dev_x, (size_t)(B*H*W*C * sizeof(float)));
	cudaMalloc((void**)&dev_X_out, (size_t)((H - K + 1)*(W - K + 1)*K*K*C * sizeof(float)));
	cudaMalloc((void**)&dev_y, (size_t)(B*M*H_out*W_out * sizeof(float)));
	cudaMalloc((void**)&dev_k, (size_t)(M*C*K*K * sizeof(float)));

	cudaMemcpy(dev_x, x, (size_t)(B*H*W*C * sizeof(float)), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_k, k, (size_t)(M*C*K*K * sizeof(float)), cudaMemcpyHostToDevice);

	//eventually make this do it in one shot
	dim3 unrollBlockDim(H*W*C, 1, 1);
	dim3 unrollGridDim(1, 1, 1);

	const int inner_dim = K*K*C;

	dim3 mulBlockDim(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 mulGridDim(ceilf((H_out*W_out)/TILE_WIDTH_FLOAT), ceilf(M/TILE_WIDTH_FLOAT), 1);
	
	//for everything in batch
	for (int b = 0; b < B; b++) {
		//unroll
		unrollKernel <<< unrollGridDim, unrollBlockDim >>> (C, H, W, K, dev_x+b*H*W*C, dev_X_out);

		//multiply first by second
		matrixMultiplyShared <<< mulGridDim, mulBlockDim >>> (dev_k, dev_X_out, dev_y+b*M*H_out*W_out, M, inner_dim, inner_dim, H_out*W_out, M, H_out*W_out);
	}

	cudaMemcpy(y, dev_y, (size_t)(B*M*H_out*W_out * sizeof(float)), cudaMemcpyDeviceToHost);

	cudaFree(dev_y);
	cudaFree(dev_x);
	cudaFree(dev_k);
	cudaFree(dev_X_out);
}

void forwardWithLoop(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
	float* dev_y;
	float* dev_x;
	float* dev_k;

	int H_out = H - K + 1;
	int W_out = W - K + 1;

	cudaMalloc((void**)&dev_x, (size_t)(B*H*W*C * sizeof(float)));
	cudaMalloc((void**)&dev_y, (size_t)(B*M*H_out*W_out * sizeof(float)));
	cudaMalloc((void**)&dev_k, (size_t)(M*C*K*K * sizeof(float)));

	cudaMemcpy(dev_x, x, (size_t)(B*H*W*C * sizeof(float)), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_k, k, (size_t)(M*C*K*K * sizeof(float)), cudaMemcpyHostToDevice);

	dim3 gridDim((B + 511) / 512);
	dim3 blockDim(512);


	forward_kernel << < gridDim, blockDim >> > (dev_y, dev_x, dev_k, B, M, C, H, W, K);

	cudaMemcpy(y, dev_y, (size_t)(B*M*H_out*W_out * sizeof(float)), cudaMemcpyDeviceToHost);

	cudaFree(dev_y);
	cudaFree(dev_x);
	cudaFree(dev_k);
}
