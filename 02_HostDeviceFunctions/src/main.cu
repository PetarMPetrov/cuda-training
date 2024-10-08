#include <cuda_runtime_api.h>
#include <iostream>

// Define a function that will only be compiled for and called from host
__host__ void HostOnly()
{
    std::cout << "This function may only be called from the host" << std::endl;
}

// Define a function that will only be compiled for and called from device
__device__ void DeviceOnly()
{
    printf("This function may only be called from the device\n");
}

// Define a function that will be compiled for both architectures
__host__ __device__ float SquareAnywhere(float x)
{
    return x * x;
}

// Call device and portable functions from a kernel
__global__ void RunGPU(float x)
{
    DeviceOnly();
    printf("%f\n", SquareAnywhere(x));
}

__host__ __device__ const char* PrintAnywhere(const char* a)
{
    return a;
}

/*
 Call host and portable functions from a kernel
 Note that, by default, if a function has no architecture
 specified, it is assumed to be __host__ by NVCC.
*/
void RunCPU(float x)
{
    HostOnly();
    std::cout << SquareAnywhere(x) << std::endl;
}

__global__ void PrintGPU()
{
    PrintAnywhere("print in gpu");
}

void PrintCPU()
{
    PrintAnywhere("print in cpu");
}

int main()
{
    std::cout << "==== Sample 02 - Host / Device Functions ====\n" << std::endl;
    /*
     Expected output:
     "This function may only be called from the host"
     1764
     "This function may only be called from the device"
     1764.00
    */

    RunCPU(42);
    RunGPU<<<1, 1>>>(42);
    PrintCPU();
    PrintGPU<<<2,2>>>();
    cudaDeviceSynchronize();
    return 0;
}

/*
Exercises:
1) Write a function that prints a message and can run on both the device and host
2) Revise the function from 1, such that the CPU version use std::cout. Use the 
__CUDA_ARCH__ macro to write code paths that contain architecture-specific code.
*/
