#include <cuda.h>
#include "cuda_runtime.h"
#include <device_functions.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>
#include <stdlib.h>

const int arraySize = 2048*4;
const int block_size = 1024;
cudaError_t sumWithCuda(float *c, float *a, unsigned int size, int type);

__global__ void sumKernelStr1(float *c, float *a){
	__shared__ float sdata[arraySize];	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = a[i];	__syncthreads();
	for (unsigned int odstep = 1; odstep < blockDim.x; odstep *= 2){
		if (tid %(2*odstep) == 0) sdata[tid] += sdata[tid + odstep];
		__syncthreads();
	}
	if (tid == 0) c[blockIdx.x] = sdata[0];
}

__global__ void sumKernelStr2(float *c, float*a) {
	__shared__ float sdata[arraySize];	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = a[i];	__syncthreads();

	for (unsigned int odstep = 1; odstep < blockDim.x; odstep *= 2) {
		int index = 2 * odstep*tid;
		if (index < blockDim.x) sdata[index] += sdata[index + odstep];
		__syncthreads();
	}

	if (tid == 0) c[blockIdx.x] = sdata[0];
}

__global__ void sumKernelStr3(float *c, float *a) {
	__shared__ float sdata[arraySize];	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = a[i];	__syncthreads();

	for (unsigned int odstep = blockDim.x / 2; odstep>0; odstep/=2) {
		if (tid < odstep) sdata[tid] += sdata[tid + odstep];
		
		__syncthreads();
	}

	if (tid == 0) c[blockIdx.x] = sdata[0];
}

int main()
{
	srand(time(NULL));
	float a[arraySize];
	for (int i = 0; i < arraySize; i++) a[i] = (float)(rand() % 20);
    float c[1] = { 0 };

    // Sum vector parallel.
	int type = 3;
    cudaError_t cudaStatus = sumWithCuda(c, a, arraySize, type);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
		getchar();
        return 1;
    }
	//for (int i = 0; i < arraySize; i++) printf("+%f", a[i]);
    printf("=%f\n",c[0]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
		getchar();
        return 1;
    }

	getchar();
    return 0;
}

// Helper function for using CUDA to sum vector in parallel.
cudaError_t sumWithCuda(float *c, float *a, unsigned int size, int type)
{
    float *dev_a = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
	cudaDeviceProp deviceProp;

	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	if (deviceProp.computeMode == cudaComputeModeProhibited){
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		getchar();
		exit(EXIT_SUCCESS);
	}
	if (cudaStatus != cudaSuccess) printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", cudaStatus, __LINE__);
	else printf("GPU Device %d: \"%s\" with compute capability %d.%d MP:%d TH_MUL:%d TH:%d WARP:%d\n\n", 0, 
		deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount, deviceProp.maxThreadsPerMultiProcessor, deviceProp.maxThreadsPerBlock, deviceProp.warpSize);
	

    // Allocate GPU buffers for 2 vectors (1 input, 1 output).
    cudaStatus = cudaMalloc((void**)&dev_c, 1 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	int threads = size;
	if(size>block_size) threads = block_size;
	//printf("%d\n",threads);
	dim3 grid(size / threads);
	// Launch a kernel on the GPU with one thread for each element.

    if(type==1)sumKernelStr1<<<grid, threads >>>(dev_c, dev_a);
	if (type == 2)sumKernelStr2 << <grid, threads >> >(dev_c, dev_a);
	if (type == 3)sumKernelStr3<< <grid, threads >> >(dev_c, dev_a);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
return cudaStatus;
}
