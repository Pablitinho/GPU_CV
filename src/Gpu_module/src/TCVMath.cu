/*
 * TCVMath.cpp
 *
 *  Created on: 19/02/2015
 *      Author: pablo
 */

#include "TCVMath.h"
#include <typeinfo>
#include "CVCudaUtils.cuh"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
using namespace std;
//==========================================================================
// Kernels
//==========================================================================
__device__ uint HammingDistance(uint x,uint y)
{

  uint dist = 0;
  uint val = x ^ y; // XOR

  // Count the number of set bits
  while(val)
  {
    ++dist;
    val &= val - 1;
  }

  return dist;
}
__global__ void Mult_HF_Kernel(half * MemSrc1, half * MemSrc2, half * MemDst, int Width, int Height)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	{
		MemDst[GlobalOffset]= __float2half(__half2float(MemSrc1[GlobalOffset])*__half2float(MemSrc2[GlobalOffset]));
	}
}
//--------------------------------------------------------------------------
__global__ void Mult_HF_Const_Kernel(half * MemSrc, float Value, half * MemDst, int Width, int Height)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	{
		MemDst[GlobalOffset]= __float2half(__half2float(MemSrc[GlobalOffset])* Value);
	}
}
//--------------------------------------------------------------------------
__global__ void Div_HF_Kernel(half * MemSrc1, half * MemSrc2, half * MemDst, int Width, int Height)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	{
		MemDst[GlobalOffset]= __float2half(__half2float(MemSrc1[GlobalOffset])/__half2float(MemSrc2[GlobalOffset]));
	}
}
//--------------------------------------------------------------------------
__global__ void Div_HF_Const_Kernel(half * MemSrc, float Value, half * MemDst, int Width, int Height)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	{
		MemDst[GlobalOffset]= __float2half(__half2float(MemSrc[GlobalOffset])/(Value));
	}
}
//--------------------------------------------------------------------------
__global__ void Subtract_HF_Kernel(half * MemSrc1, half * MemSrc2, half * MemDst, int Width, int Height)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	{
		MemDst[GlobalOffset]= __float2half(__half2float(MemSrc1[GlobalOffset])-__half2float(MemSrc2[GlobalOffset]));
	}
}
//--------------------------------------------------------------------------
__global__ void SubtractAbs_Kernel(half * MemSrc1, half * MemSrc2, half * MemDst, int Width, int Height)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	{
		MemDst[GlobalOffset]= __float2half(abs(__half2float(MemSrc1[GlobalOffset])-__half2float(MemSrc2[GlobalOffset])));
	}
}
//--------------------------------------------------------------------------
__global__ void Subtract_HF_Const_Kernel(half * MemSrc, float Value, half * MemDst, int Width, int Height)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	{
		MemDst[GlobalOffset]= __float2half(__half2float(MemSrc[GlobalOffset])-(Value));
	}
}
//--------------------------------------------------------------------------
__global__ void Subtract_Kernel(unsigned char * MemSrc1, unsigned char * MemSrc2, half * MemDst, int Width, int Height)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	{
		MemDst[GlobalOffset]= __float2half( (float)(MemSrc1[GlobalOffset]-MemSrc2[GlobalOffset]) );
	}
}
//--------------------------------------------------------------------------
__global__ void HammingDistance_Kernel(unsigned int * MemSrc1,unsigned int * MemSrc2,half * MemDst,float Factor,int Width,int Height)
{
   //------------------------------------------------------------------
   int globalX = blockIdx.x * blockDim.x + threadIdx.x;
   int globalY = blockIdx.y * blockDim.y + threadIdx.y;

   int GlobalOffset = (globalY * Width + globalX);
   //------------------------------------------------------------------
   uint Value1;
   uint Value2;
   float HD=0;
   //===============================================================================================
	if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
   {
	   Value1=(uint)MemSrc1[GlobalOffset];
	   Value2=(uint)MemSrc2[GlobalOffset];

	   HD = (float)HammingDistance(Value1,Value2);

	   MemDst[GlobalOffset] = __float2half(HD*Factor);
   }
}
//--------------------------------------------------------------------------
__global__ void Transpose_Kernel(half * MemSrc,half * MemDst,int Width,int Height)
{
	//------------------------------------------------------------------
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    int GlobalOffsetT = (globalX * Height + globalY);

	if (globalX>=0 && globalX<Width && globalY>=0 && globalY<Height)
    {
	    MemDst[GlobalOffsetT]=MemSrc[GlobalOffset];
    }
}
//--------------------------------------------------------------------------
__global__ void Divergence_Kernel(half * MemIm1, half * MemIm2,half * MemOut,int Width,int Height)
{
   //===============================================================================================
   //
   //===============================================================================================
   int globalX = blockIdx.x * blockDim.x + threadIdx.x;
   int globalY = blockIdx.y * blockDim.y + threadIdx.y;

   int GlobalOffset = (globalY * Width + globalX);
   //===============================================================================================
   if (globalX>2 && globalY>2 && globalX<Width-3 && globalY<Height-3)
   {
	   //----------------------------------------------------------------
	   float u_x=0,u_y=0,E=0,N=0,Center1=0,Center2=0;
       
	   //----------------------------------------------------------------
	   // u_x u_y
	   //----------------------------------------------------------------
	   E=0.0f;
	   N=0.0f;
	   Center1=__half2float(MemIm1[GlobalOffset]);
	   Center2=__half2float(MemIm2[GlobalOffset]);

	   E=__half2float(MemIm1[GlobalOffset+1]);
	   N=__half2float(MemIm2[GlobalOffset-Width]);

	   u_x=Center1-E;
	   u_y=Center2-N;
	   float Result = u_x+u_y;
	   //----------------------------------------------------------------
	   MemOut[GlobalOffset]=__float2half(Result);

   }
   else
   {
		if (globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
		{
			MemOut[GlobalOffset]= __float2half(0.0f);
		}

   }
}
//==========================================================================
__global__ void FloatMaxMinAvg_Kernel(half * MemMax,half * MemMin,half * MemAvg,half * MemMaxOut,half * MemMinOut,half * MemAvgOut, int Size)
{
	//------------------------------------------------------------------
	int GlobalOffset = blockIdx.x * blockDim.x + threadIdx.x;
	int LocalOffset = threadIdx.x;

	extern __shared__ float s[];
	float *MaxCache = s;
	float *MinCache = (float*)&MaxCache[blockDim.x];
	float *AvgCache = (float*)&MinCache[blockDim.x];
	//------------------------------------------------------------------
    if (GlobalOffset<Size)
    {
        MaxCache[LocalOffset] =__half2float(MemMax[GlobalOffset]);
    	MinCache[LocalOffset] =__half2float(MemMin[GlobalOffset]);
        AvgCache[LocalOffset] =__half2float(MemAvg[GlobalOffset]);

        __syncthreads();

    	int nTotalThreads = blockDim.x;// get_local_size(0);	// Total number of active threads
    	float temp=0.0f;
    	uint halfPoint = (nTotalThreads);

    	while(nTotalThreads >1)
    	{
    		//int halfPoint = (nTotalThreads >> 1);	// divide by two
    	    halfPoint = (uint)ceil((float)halfPoint / 2.0f);	// divide by two
    		// only the first half of the threads will be active.
    		if (LocalOffset < halfPoint)
    		{
    			// Get the shared value stored by another thread
    			temp = MinCache[LocalOffset + halfPoint];
    		    if (temp < MinCache[LocalOffset] && !isnan(temp))
    			{
    				MinCache[LocalOffset] = temp;
    			}
    			temp = MaxCache[LocalOffset + halfPoint];
    			if (temp > MaxCache[LocalOffset]&& !isnan(temp))
    			{
    				MaxCache[LocalOffset] = temp;
    			}
    		    // when calculating the average, sum and divide
    			if (!isnan(AvgCache[LocalOffset + halfPoint])){
    				AvgCache[LocalOffset] += AvgCache[LocalOffset + halfPoint];
    				AvgCache[LocalOffset] /= 2;
    			}
    		}
    		__syncthreads();
    		if (nTotalThreads!=1)
    	    	nTotalThreads = (uint)((float)nTotalThreads /2.0f);	// divide by two.
    		else
    			nTotalThreads=0;
    	}

    	if (LocalOffset == 0)
    	{
    		if (!isnan(MaxCache[0]))
    			MemMaxOut[blockIdx.x] = __float2half(MaxCache[0]);

    		if (!isnan(MinCache[0]))
    			MemMinOut[blockIdx.x] = __float2half(MinCache[0]);

    		if (!isnan(AvgCache[0]))
    			MemAvgOut[blockIdx.x] = __float2half(AvgCache[0]);
    	}
    }
}
//==========================================================================
__global__ void Norm_Kernel(half * MemSrc1, half * MemSrc2, half * MemDst, int Width, int Height)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	{
		float fx=__half2float(MemSrc1[GlobalOffset]);
		float fy=__half2float(MemSrc2[GlobalOffset]);
		MemDst[GlobalOffset]= __float2half(sqrt(fx*fx+fy*fy));
	}
}
//==========================================================================
__global__ void Range_Kernel(half * MemSrc, half * MemDst, float MaxIn,float MinIn,float MaxOut,float MinOut,int Width, int Height)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	{
		float Value=__half2float(MemSrc[GlobalOffset]);
		Value= Range_Value(Value, MaxIn, MinIn,MaxOut,MinOut);

		MemDst[GlobalOffset]= __float2half(Value);
	}
}
//==========================================================================
__global__ void Abs_Kernel(half * MemSrc, half * MemDst, int Width, int Height)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if(globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
	{
		MemDst[GlobalOffset]= __float2half(abs(__half2float(MemSrc[GlobalOffset])));
	}
}
//--------------------------------------------------------------------------

// End Kernels
//==========================================================================
//--------------------------------------------------------------------------
//==========================================================================
// Class Methods
//==========================================================================
TCVMath::TCVMath(void * d_Gpu)
{
	Gpu = d_Gpu;
}
//--------------------------------------------------------------------------
void TCVMath::Mult(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc1->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc1->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Mult_HF_Kernel<<<numBlocks, numThreads>>>(MemSrc1->GetMemory(),MemSrc2->GetMemory(), MemDst->GetMemory(), MemSrc1->Width(), MemSrc1->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMath::Mult(TGpuMem::TGpuMemHalfFloat * MemSrc,float Value,TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Mult_HF_Const_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(), Value, MemDst->GetMemory(), MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMath::Div(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc1->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc1->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Div_HF_Kernel<<<numBlocks, numThreads>>>(MemSrc1->GetMemory(),MemSrc2->GetMemory(), MemDst->GetMemory(), MemSrc1->Width(), MemSrc1->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMath::Div(TGpuMem::TGpuMemHalfFloat * MemSrc,float Value,TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Div_HF_Const_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(),Value, MemDst->GetMemory(), MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMath::Subtract(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc1->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc1->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Subtract_HF_Kernel<<<numBlocks, numThreads>>>(MemSrc1->GetMemory(),MemSrc2->GetMemory(), MemDst->GetMemory(), MemSrc1->Width(), MemSrc1->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMath::Eucliden_Norm(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc1->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc1->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Norm_Kernel<<<numBlocks, numThreads>>>(MemSrc1->GetMemory(),MemSrc2->GetMemory(), MemDst->GetMemory(), MemSrc1->Width(), MemSrc1->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMath::Subtract(TGpuMem::TGpuMemHalfFloat * MemSrc,float Value,TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Subtract_HF_Const_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(),Value, MemDst->GetMemory(), MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMath::Subtract(TGpuMem::TGpuMemUChar * MemSrc1,TGpuMem::TGpuMemUChar * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc1->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc1->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Subtract_Kernel<<<numBlocks, numThreads>>>(MemSrc1->GetMemory(),MemSrc2->GetMemory(), MemDst->GetMemory(), MemSrc1->Width(), MemSrc1->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMath::SubtractAbs(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc1->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc1->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    SubtractAbs_Kernel<<<numBlocks, numThreads>>>(MemSrc1->GetMemory(),MemSrc2->GetMemory(), MemDst->GetMemory(), MemSrc1->Width(), MemSrc1->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMath::HammingDistance(TGpuMem::TGpuMemUInt * MemSrc1,TGpuMem::TGpuMemUInt * MemSrc2, TGpuMem::TGpuMemHalfFloat * MemDst,float Factor)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc1->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc1->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    HammingDistance_Kernel<<<numBlocks, numThreads>>>(MemSrc1->GetMemory(), MemSrc2->GetMemory(),MemDst->GetMemory(), Factor,MemSrc1->Width(), MemSrc1->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMath::Transpose(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Transpose_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(), MemDst->GetMemory(), MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMath::Divergence(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc1->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc1->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Divergence_Kernel<<<numBlocks, numThreads>>>(MemSrc1->GetMemory(),MemSrc2->GetMemory(), MemDst->GetMemory(), MemSrc1->Width(), MemSrc1->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMath::MaxMinAvg(TGpuMem::TGpuMemHalfFloat * MemSrc,float &Max,float &Min,float &Avg)
{
     //----------------------------------------------
     // Mem Aux
     //----------------------------------------------
	 TGpuMem::TGpuMemHalfFloat *MaxMem=NULL;
	 TGpuMem::TGpuMemHalfFloat *MinMem=NULL;
	 TGpuMem::TGpuMemHalfFloat *AvgMem=NULL;

	 TGpuMem::TGpuMemHalfFloat *MaxMemAux=NULL;
	 TGpuMem::TGpuMemHalfFloat *MinMemAux=NULL;
	 TGpuMem::TGpuMemHalfFloat *AvgMemAux=NULL;
     //----------------------------------------------
     float pp1 = (float)(log(MemSrc->Width()) / log(2));
     float pp = (float) ceil(log(MemSrc->Width())/ log(2));

     uint MemAuxWidth = (uint)ceil(log(MemSrc->Width()) / log(2));
     uint MemAuxHeight= (uint)ceil(log(MemSrc->Height()) / log(2));

     //TGpuMem::TGpuMemHalfFloat *MemoryInputAux = new TGpuMem::TGpuMemHalfFloat(Gpu, (uint)pow(2.0, (double)MemAuxWidth),(uint)pow(2.0, (double)MemAuxHeight),1);

     TGpuMem::TGpuMemHalfFloat *MemoryInputAux = new TGpuMem::TGpuMemHalfFloat(Gpu, MemSrc->Width(),MemSrc->Height(),1, false);

     MemSrc->Copy(MemSrc);

     ((TGpu *)Gpu)->CV->Utils->Padding(MemSrc,MemoryInputAux,0,0,sqrt(-1));
     //----------------------------------------------
     uint MemSize=0,MemSizeAnt = 0;
     bool exit = false;
     bool FirstTime = true;
     dim3 numThreads;
     dim3 numBlocks;
     long BlockSize = MaxFactor(MemSrc->Width() * MemSrc->Height(), 128);
     while (!exit)
     {
         if (FirstTime)
         {
             //----------------------------------------------
             // Set Global and Local Work Size
             //----------------------------------------------
             MemSize = (uint)((MemoryInputAux->Width() * MemoryInputAux->Height()) / (BlockSize));

             numThreads = dim3(BlockSize, 1, 1);
             numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemoryInputAux->Size(), numThreads.x), 1, numThreads.y);
         }
         else
         {
             //----------------------------------------------
             // Set Global and Local Work Size
             //----------------------------------------------
             numThreads = dim3(BlockSize, 1, 1);
             numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSizeAnt, numThreads.x), 1, numThreads.y);
         }
         MaxMem = new TGpuMem::TGpuMemHalfFloat(Gpu, MemSize, 1,1, false);
         MinMem = new TGpuMem::TGpuMemHalfFloat(Gpu, MemSize, 1,1, false);
         AvgMem = new TGpuMem::TGpuMemHalfFloat(Gpu, MemSize, 1,1, false);
         //----------------------------------------------
         // Set Arguments
         //----------------------------------------------
         if (FirstTime)
         {
          	//----------------------------------------------------------------------------------------------------
          	// Estimate the number of Blocks and number Threads
          	//----------------------------------------------------------------------------------------------------
        	 FloatMaxMinAvg_Kernel<<<numBlocks, numThreads,BlockSize*3*sizeof(float)>>>(MemSrc->GetMemory(),MemSrc->GetMemory(),MemSrc->GetMemory(),MaxMem->GetMemory(),MinMem->GetMemory(),AvgMem->GetMemory(),MemSrc->Size());
             cudaThreadSynchronize();
         }
         else
         {
        	 FloatMaxMinAvg_Kernel<<<numBlocks, numThreads,BlockSize*3*sizeof(float)>>>(MaxMemAux->GetMemory(),MinMemAux->GetMemory(),AvgMemAux->GetMemory(),MaxMem->GetMemory(),MinMem->GetMemory(),AvgMem->GetMemory(),MemSrc->Size());
             cudaThreadSynchronize();
         }

         MemSizeAnt = MemSize;
         MemSize = MemSize / (uint)(BlockSize);
         BlockSize = MaxFactor(MemSizeAnt, 128);

         if (MemSizeAnt % (BlockSize) != 0 || MemSize==0) //MemSizeAnt < (uint)(BlockSize) ||
         {
             exit = true;
         }
         else
         {
             if (!FirstTime)
             {
            	 delete MaxMemAux;
            	 delete MinMemAux;
            	 delete AvgMemAux;
             }

             MaxMemAux = new TGpuMem::TGpuMemHalfFloat(Gpu, MemSizeAnt, 1,1, false);
             MinMemAux = new TGpuMem::TGpuMemHalfFloat(Gpu, MemSizeAnt, 1,1, false);
             AvgMemAux = new TGpuMem::TGpuMemHalfFloat(Gpu, MemSizeAnt, 1,1, false);

             MaxMem->Copy(MaxMemAux);
             MinMem->Copy(MinMemAux);
             AvgMem->Copy(AvgMemAux);

             delete MaxMem;
             delete MinMem;
             delete AvgMem;
             FirstTime = false;
         }
     }
     //----------------------------------------------
     delete MemoryInputAux;
     //----------------------------------------------
     float *MaxTemp;
     float *MinTemp;
     float *AvgTemp;

     MaxTemp=MaxMem->CopyFromDevice();
     MinTemp=MinMem->CopyFromDevice();
     AvgTemp=AvgMem->CopyFromDevice();

     Max = MaxTemp[0];
     Min = MinTemp[0];
     Avg = AvgTemp[0];

     float MaxT,MinT,AvgT;
     for (uint i = 1; i < MaxMem->Size(); i++)
     {

          MaxT = MaxTemp[i];
          MinT = MinTemp[i];
          AvgT = AvgTemp[i];

          if (Max < MaxT)
              Max = MaxT;

          if (Min > MinT)
              Min = MinT;

          Avg += AvgT;
     }
     Avg = Avg / MaxMem->Size();
     //----------------------------------------------
     //Dispose
     //----------------------------------------------
     if (MaxMem != NULL)
     {
         delete MaxMem;
         delete MinMem;
         delete AvgMem;
     }
     if (MaxMemAux != NULL)
     {
    	 delete MaxMemAux;
    	 delete MinMemAux;
    	 delete AvgMemAux;
     }
     //----------------------------------------------

}
//--------------------------------------------------------------------------
unsigned int TCVMath::MaxFactor(unsigned int number,unsigned int Maximum)
{
     uint NumMax = 0;
     uint maxi = (uint)sqrt(number);  //round down
     maxi = max(Maximum,maxi);
     for (uint factor = 1; factor <= maxi; ++factor)
     {
         if (number % factor == 0)
         {
             if (factor > NumMax && factor <= Maximum)
             {
                 NumMax = factor;
             }
             if (factor != maxi)
             {
                 if (factor > NumMax && factor <= Maximum)
                 {
                     NumMax = factor;
                 }
             }
         }
     }
     return NumMax;
}
//--------------------------------------------------------------------------
int TCVMath::MaximumScales(uint Width, uint Height, int NumMax,int MinSize, float Factor)
{
    int NumPir = 1;
    for (int i = 1; i < NumMax; i++)
    {
         if ((Width * Factor) <= MinSize || (Height * Factor) <= MinSize)
         {
              i = NumMax + 1;
         }
         else NumPir++;

         Width = (uint)(Width * Factor);
         Height = (uint)(Height * Factor);
    }
    return min(NumMax, NumPir);
}
//--------------------------------------------------------------------------
void TCVMath::Range(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat *MemDst,float MaxOut, float MinOut)
{
    float Max,Min,Avg;
    ((TGpu *)Gpu)->CV->Math->MaxMinAvg(MemSrc,Max,Min,Avg);

	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Range_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(),MemDst->GetMemory(),Max,Min,MaxOut,MinOut,MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVMath::Abs(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat *MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Abs_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(),MemDst->GetMemory(), MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();

}

TCVMath::~TCVMath()
{

}
//--------------------------------------------------------------------------

