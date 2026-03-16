#include <cuda_runtime.h>
#include "../include/activation.h"

__global__ void relu(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n)
        data[i] = max(0.0f, data[i]);
}

void launch_relu(float *data, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    relu<<<blocks, threads>>>(data, n);
}