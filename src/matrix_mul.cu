#include <cuda_runtime.h>
#include <iostream>
#include "matrix.h"

__global__ void matmulKernel(float* A, float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N)
    {
        float sum = 0.0f;

        for(int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

void gpuMatrixMultiply(float* A, float* B, float* C, int N)
{
    float *d_A, *d_B, *d_C;

    size_t size = N * N * sizeof(float);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16,16);
    dim3 gridSize((N+15)/16,(N+15)/16);

    matmulKernel<<<gridSize,blockSize>>>(d_A,d_B,d_C,N);

    cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}