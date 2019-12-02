#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
__global__ void forward_kernel1(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP\
  //__device__ __shared__ int TILE_WIDTH = 32;
  __shared__ float subTileA[32][32];
  __shared__ float subTileB[32][32];
  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  int Row = by * 32 + ty;
  int Col = bx * 32 + tx;
  float Pvalue = 0;
  
  int tiles = numAColumns/32;
  if (numAColumns%32) tiles++;
  
  for (int m=0; m<tiles; ++m){
    if((Row < numARows) && ((m*32+tx)<numAColumns)) {
      subTileA[ty][tx] = A[Row*numAColumns + m*32 + tx];
    }
    else{
      subTileA[ty][tx] = 0;
    }
    if ((Col < numBColumns) && ((m*32 + ty)<numBRows)){
      subTileB[ty][tx] = B[(m*32+ty)*numBColumns + Col];
    }
    else {
      subTileB[ty][tx] = 0;
    }
      __syncthreads();
      for (int k = 0;k<32;++k){
        Pvalue += subTileA[ty][k] * subTileB[k][tx];
      }
      __syncthreads();
    }
  if (Row < numCRows && Col < numCColumns){
  C[Row*numCColumns+Col] = Pvalue;
  }
}

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
void unroll(int C, int H, int W, int K, float* X, float* X_unrolled){
    int H_out = H-K+1;
    int W_out = W-K+1;
    for (int c=0; c<C; ++c){
        for (int p =0; p<K; p++){
            for (int q=0; q<K; q++){
                for (int h=0; h<H_out; h++){
                    for (int w=0; w<W_out; w++){
                        int unroll_col_index = h*W_out + w;
                        int unroll_row_index = c*K*K + p*K+ q;
                        X_unrolled[unroll_row_index*H_out*W_out+unroll_col_index] = X[c*H*W+(h+p)*W+(w+q)];
                    }
                }
            }
        }
    }
}
/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   We only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // // Use mxnet's CHECK_EQ to do assertions.
    // CHECK_EQ(0, 1) 

    const int B = x.shape_[0];
    const int M = y.shape_[1]; // num_filter
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    //unroll X
    const int W_out = W-K+1;
    const int H_out = H-K+1;
    const int W_unroll = C*K*K;
    const int H_unroll = H_out * W_out;
    float* X_unroll = (float*)malloc(W_unroll * H_unroll * sizeof(float));
    for (int b=0; b<B; b++){
    unroll(C, H, W, K, x.dptr_+b*C*H*W, X_unroll);
    printf
    dim3 gridDim(ceil(H_unroll/32),ceil(M/32));
    dim3 blockDim(32,32);

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    forward_kernel1<<<gridDim,blockDim>>>(w.dptr_, X_unroll, y.dptr_+b*M*H_out*W_out, M, C*K*K, C*K*K, H_out*W_out, M, H_out*W_out);
 	   //forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    }
    free(X_unroll);
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    assert(0 && "No forward implementation for other datatypes needed");
}
}
}

#endif
