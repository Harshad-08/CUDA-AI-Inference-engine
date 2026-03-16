#include <cuda_runtime.h>

__global__ void relu(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        if(data[i] < 0)
            data[i] = 0;
    }
}