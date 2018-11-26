//--------------------------------------------------------------------------
//   This Class handles the GPU memory
//   Created on: 18/02/2015
//   Author: Pablo Guzman

//--------------------------------------------------------------------------
#include "TGpu.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "defines.h"

using namespace std;
//--------------------------------------------------------------------------
TGpu::TGpu(int Block_X,int Block_Y)
{
	BlockX = Block_X;
    BlockY = Block_Y;
	CV = new TCV(this);

	support_zero_copy = false;

	InitTimer();
}
//--------------------------------------------------------------------------
bool TGpu::Allow_zero_copy() 
{
	bool success = false;

	cudaDeviceProp prop;

	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, this->GetDevice()));

	if (prop.canMapHostMemory == 1) 
	{
		CUDA_CHECK_RETURN(cudaSetDeviceFlags(cudaDeviceMapHost));
		success = true;
		support_zero_copy = true;
	}
	else 
	{
		printf("Function not supported \n");
		success = false;
		support_zero_copy = false;
	}

	return success;
}
//--------------------------------------------------------------------------
int TGpu::CountGPUs()
{
    int count;
    CUDA_CHECK_RETURN(cudaGetDeviceCount( &count ));
    return count;
}
//--------------------------------------------------------------------------
void TGpu::PrintProperties(int Device_Num)
{
	cudaDeviceProp prop;

	CUDA_CHECK_RETURN(cudaGetDeviceProperties( &prop, Device_Num ));

	cout << "--- General Information for device ... " << Device_Num << endl;
	cout << "Name: " << prop.name << endl;
	cout << "Compute capability: " << prop.major << " "<< prop.minor << endl;
	cout << "Clock rate: " << prop.clockRate << endl;
	cout << "Device copy overlap: ";
	if (prop.deviceOverlap) cout << "Enabled" << endl;
	else cout << "Disabled" << endl;
	cout << "Kernel execition timeout : ";
	if (prop.kernelExecTimeoutEnabled) cout << "Enabled" << endl;
	else cout << "Disabled" << endl;
	cout << "--- Memory Information for device "<< Device_Num << endl;
	cout << "Total global mem: " << prop.totalGlobalMem << endl;
	cout << "Total constant Mem: " << prop.totalConstMem << endl;
	cout << "Max mem pitch: " << prop.memPitch << endl;
	cout << "Texture Alignment: "<< prop.textureAlignment << endl;
	cout << "--- MP Information for device " << Device_Num << endl;
	cout << "Multiprocessor count: " <<prop.multiProcessorCount << endl;
	cout << "Shared mem per mp: " << prop.sharedMemPerBlock << endl;
	cout << "Registers per mp: " << prop.regsPerBlock << endl;
	cout << "Threads in warp: " << prop.warpSize << endl;
	cout << "Max threads per block: " <<prop.maxThreadsPerBlock << endl;
	cout << "Max thread dimensions: (" << prop.maxThreadsDim[0] << ", "<< prop.maxThreadsDim[1]<< ", " << prop.maxThreadsDim[2] << ")"<< endl;
	cout << "Max grid dimensions: ("<<prop.maxGridSize[0]<< ", " << prop.maxGridSize[1] << ", " <<prop.maxGridSize[2] << ")" << endl;

}
//--------------------------------------------------------------------------
void TGpu::GetLastError()
{
	CUDA_CHECK_RETURN(cudaGetLastError());
}
//--------------------------------------------------------------------------
void TGpu::SetDevice(int DevNum)
{
	CUDA_CHECK_RETURN(cudaSetDevice(DevNum));
}
//--------------------------------------------------------------------------
int TGpu::GetDevice()
{
	int device_id;
	CUDA_CHECK_RETURN(cudaGetDevice(&device_id));

	return device_id;
}
//--------------------------------------------------------------------------
void TGpu::ResetDevice()
{
	CUDA_CHECK_RETURN(cudaDeviceReset());
}
//--------------------------------------------------------------------------
int TGpu::iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}
//--------------------------------------------------------------------------
void TGpu::SetBlockSize(int Block_X,int Block_Y)
{
	BlockX = Block_X;
    BlockY = Block_Y;
}
//--------------------------------------------------------------------------
int TGpu::GetBlockX() { return BlockX; }
//--------------------------------------------------------------------------
int TGpu::GetBlockY() { return BlockY; }
//--------------------------------------------------------------------------
void TGpu::InitTimer()
{
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
}
//--------------------------------------------------------------------------
void TGpu::StartMeasurement()
{
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
}
//--------------------------------------------------------------------------
float TGpu::StopMeasurement()
{
	float elapsed;
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsed, start, stop));

	return elapsed;
}
//--------------------------------------------------------------------------
void TGpu::DestroyTimer()
{
	CUDA_CHECK_RETURN(cudaEventDestroy(start));
	CUDA_CHECK_RETURN(cudaEventDestroy(stop));
}
//--------------------------------------------------------------------------
void TGpu::SetCacheConfig(CacheConfig config)
{
	cudaFuncCache Config_device = (cudaFuncCache)config;
	CUDA_CHECK_RETURN(cudaDeviceSetCacheConfig(Config_device));
}
//--------------------------------------------------------------------------
void TGpu::GetMemoryInfo(size_t * FreeMemory, size_t *TotalMemory)
{
	CUDA_CHECK_RETURN(cudaMemGetInfo(FreeMemory,TotalMemory));
}
//--------------------------------------------------------------------------
TGpu::~TGpu()
{
	DestroyTimer();
	delete CV;
	ResetDevice();
}
//--------------------------------------------------------------------------

