#include <cuda_runtime_api.h>
#include <iostream>

__constant__ int a;

__global__ void DoNothing()
{
    int ba = a;
    printf("%d", ba);
}

__device__ const int a = 10;

int main()
{
    return 0;
}
