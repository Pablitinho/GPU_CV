/*
 * TCVFilters.cpp
 *
 *  Created on: 23/02/2015
 *      Author: pablo
 */

#include "TCVFilters.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <typeinfo>
#include <stdio.h>
#include "CVCudaUtils.cuh"
using namespace std;
//#include "CVCudaUtils.cuh"
//==========================================================================

// Kernels

//===============================================================================================
__device__ uint HammingDistance2(uint x,uint y)
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
//==========================================================================
__global__ void Median3x3_Kernel(unsigned short * MemSrc, unsigned short * MemDst, int Width, int Height)
{

	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int OffsetIm = (globalY * Width + globalX);

	/*
    int threadX = threadIdx.x+1;
    int threadY = threadIdx.y+1;
    int blockDimX = blockDim.x+2;
    */
    //int blockDimY = blockDim.y+2;

    //int OffsetLocal = (threadY * blockDimX + threadX);

    extern __shared__ float DataCache[];
    //printf("[%d]: shared value is %d\n", threadIdx.x, DataCache[OffsetLocal]);
	//FillCache(DataCache, MemSrc, 1, Width, Height);
    //------------------------------------------------------------------
    if (globalX>0 && globalX<(Width-1) && globalY>0 && globalY<(Height-1))
	{
	        float r0,r1,r2,r3,r4,r5,r6,r7,r8;

	        r0=__half2float(MemSrc[OffsetIm]);
	        r1=__half2float(MemSrc[OffsetIm-1]);
	        r2=__half2float(MemSrc[OffsetIm+1]);
	        r3=__half2float(MemSrc[OffsetIm+Width]);
	        r5=__half2float(MemSrc[OffsetIm+Width+1]);
	        r8=__half2float(MemSrc[OffsetIm+Width-1]);
	        r4=__half2float(MemSrc[OffsetIm-Width]);
	        r6=__half2float(MemSrc[OffsetIm-Width-1]);
	        r7=__half2float(MemSrc[OffsetIm-Width+1]);

/*
	        r0=(DataCache[OffsetLocal]);
	        r1=(DataCache[OffsetLocal-1]);
	        r2=(DataCache[OffsetLocal+1]);
	        r3=(DataCache[OffsetLocal+blockDimX]);
	        r4=(DataCache[OffsetLocal-blockDimX]);
	        r5=(DataCache[OffsetLocal+blockDimX+1]);
	        r6=(DataCache[OffsetLocal-blockDimX-1]);
	        r7=(DataCache[OffsetLocal-blockDimX+1]);
	        r8=(DataCache[OffsetLocal+blockDimX-1]);
*/
	        //perform partial bitonic sort to find current channel median

	        float uiMin = min(r0, r1);
	        float uiMax = max(r0, r1);
	        r0 = uiMin;
	        r1 = uiMax;

	        uiMin = min(r3, r2);
	        uiMax = max(r3, r2);
	        r3 = uiMin;
	        r2 = uiMax;

	        uiMin = min(r2, r0);
	        uiMax = max(r2, r0);
	        r2 = uiMin;
	        r0 = uiMax;

	        uiMin = min(r3, r1);
	        uiMax = max(r3, r1);
	        r3 = uiMin;
	        r1 = uiMax;

	        uiMin = min(r1, r0);
	        uiMax = max(r1, r0);
	        r1 = uiMin;
	        r0 = uiMax;

	        uiMin = min(r3, r2);
	        uiMax = max(r3, r2);
	        r3 = uiMin;
	        r2 = uiMax;

	        uiMin = min(r5, r4);
	        uiMax = max(r5, r4);
	        r5 = uiMin;
	        r4 = uiMax;

	        uiMin = min(r7, r8);
	        uiMax = max(r7, r8);
	        r7 = uiMin;
	        r8 = uiMax;

	        uiMin = min(r6, r8);
	        uiMax = max(r6, r8);
	        r6 = uiMin;
	        r8 = uiMax;

	        uiMin = min(r6, r7);
	        uiMax = max(r6, r7);
	        r6 = uiMin;
	        r7 = uiMax;

	        uiMin = min(r4, r8);
	        uiMax = max(r4, r8);
	        r4 = uiMin;
	        r8 = uiMax;

	        uiMin = min(r4, r6);
	        uiMax = max(r4, r6);
	        r4 = uiMin;
	        r6 = uiMax;

	        uiMin = min(r5, r7);
	        uiMax = max(r5, r7);
	        r5 = uiMin;
	        r7 = uiMax;

	        uiMin = min(r4, r5);
	        uiMax = max(r4, r5);
	        r4 = uiMin;
	        r5 = uiMax;

	        uiMin = min(r6, r7);
	        uiMax = max(r6, r7);
	        r6 = uiMin;
	        r7 = uiMax;

	        uiMin = min(r0, r8);
	        uiMax = max(r0, r8);
	        r0 = uiMin;
	        r8 = uiMax;

	        r4 = max(r0, r4);
	        r5 = max(r1, r5);

	        r6 = max(r2, r6);
	        r7 = max(r3, r7);

	        r4 = min(r4, r6);
	        r5 = min(r5, r7);

	        //store found median into result
	        MemDst[OffsetIm]= __float2half_rn(min(r4, r5));
		   //===============================================================================================
	}
    else
	{
      if (globalX>=0 && globalX<Width && globalY>=0 && globalY<Height)
		  MemDst[OffsetIm]=MemSrc[OffsetIm];
	}
}
//==========================================================================
__global__ void Separable_Convolution_H_Kernel(unsigned short * MemSrc, unsigned short * MemDst,float * MemFilter,int FilterSize, int Width, int Height)
{
	//------------------------------------------------------------------
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);

    int LocalOffset = (threadIdx.x * blockDim.x + threadIdx.y);
	//------------------------------------------------------------------
    int HalfFilter = floor((float)FilterSize/2.0);

	extern __shared__ float FilterCache[];
	//===============================================================================================
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
	{

 	   if (LocalOffset<FilterSize)
 	   {
 		  FilterCache[LocalOffset]=MemFilter[LocalOffset];
 	   }
 	   __syncthreads();

	   float Value;
	   //===============================================================================================
	   //
	   //===============================================================================================
	   int Pos=0;
	   float Result=0.0f;

	   for (int i=0;i<FilterSize;i++)
	   {
		    Pos = globalY * Width + globalX + (i-HalfFilter);

			if ((globalX+(i-HalfFilter)) >= 0 && (globalX+(i-HalfFilter)) < Width)
			{
				Value =	__half2float(MemSrc[Pos])*FilterCache[i];
	  		    Result=Result+Value;
			}
			else
			{
			    Value = __half2float(MemSrc[GlobalOffset])*FilterCache[i];
			    Result=Result+Value;
			}
	   }
	   MemDst[GlobalOffset]=__float2half_rn(Result);
	 }
}
//==========================================================================
__global__ void Separable_Convolution_V_Kernel(unsigned short * MemSrc, unsigned short * MemDst,float * MemFilter,int FilterSize, int Width, int Height)
{
	//------------------------------------------------------------------
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);

    int LocalOffset = (threadIdx.x * blockDim.x + threadIdx.y);
	//------------------------------------------------------------------
    int HalfFilter = floor((float)FilterSize/2.0);

	extern __shared__ float FilterCache[];
	//===============================================================================================
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
	{
 	   if (LocalOffset<FilterSize)
 	   {
 		  FilterCache[LocalOffset]=MemFilter[LocalOffset];
 	   }
 	   __syncthreads();

	   float Value;
	   //===============================================================================================
	   //
	   //===============================================================================================
	   int Pos=0;
	   float Result=0.0f;

	   for (int i=0;i<FilterSize;i++)
	   {
		    Pos = (i-HalfFilter+globalY) * Width  + globalX ;

			if ((globalY+(i-HalfFilter)) >= 0 && (globalY+(i-HalfFilter)) < Height)
			{
				Value =	__half2float(MemSrc[Pos])*FilterCache[i];
	  		    Result=Result+Value;
			}
			else
			{
			    Value = __half2float(MemSrc[GlobalOffset])*FilterCache[i];
			    Result=Result+Value;
			}
	   }
	   MemDst[GlobalOffset]=__float2half_rn(Result);
	 }
}
//==========================================================================
//==========================================================================
__global__ void Separable_Convolution_H_Kernel(unsigned char * MemSrc, unsigned short * MemDst,float * MemFilter,int FilterSize, int Width, int Height)
{
	//------------------------------------------------------------------
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    int LocalOffset = (threadIdx.x * blockDim.x + threadIdx.y);
	//------------------------------------------------------------------
    int HalfFilter = (FilterSize/2);

	extern __shared__ float FilterCache[];
	//===============================================================================================
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
	{
 	   if (LocalOffset<FilterSize)
 	   {
 		  FilterCache[LocalOffset]=MemFilter[LocalOffset];
 	   }
 	   __syncthreads();

	   float Value;
	   //===============================================================================================
	   //
	   //===============================================================================================
	   int Pos=0;
	   float Result=0.0f;

	   for (int i=0;i<FilterSize;i++)
	   {
		    Pos = globalY * Width + globalX + (i-HalfFilter);

			if (globalX+(i-HalfFilter) >= 0 && globalX+(i-HalfFilter) < Width)
			{
				Value =	((float)MemSrc[Pos])*FilterCache[i];
	  		    Result=Result+Value;
			}
			else
			{
			    Value = ((float)MemSrc[GlobalOffset])*FilterCache[i];
			    Result=Result+Value;
			}
	   }
	   MemDst[GlobalOffset]=__float2half_rn(Result);
	 }
}
//==========================================================================
__global__ void Separable_Convolution_V_Kernel(unsigned char * MemSrc, unsigned short * MemDst,float * MemFilter,int FilterSize, int Width, int Height)
{
	//------------------------------------------------------------------
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);

    int LocalOffset = (threadIdx.x * blockDim.x + threadIdx.y);
	//------------------------------------------------------------------
    int HalfFilter = (FilterSize/2);

	extern __shared__ float FilterCache[];
	//===============================================================================================
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
	{

 	   if (LocalOffset<FilterSize)
 	   {
 		  FilterCache[LocalOffset]=MemFilter[LocalOffset];
 	   }
 	   __syncthreads();

	   float Value;
	   //===============================================================================================
	   //
	   //===============================================================================================
	   int Pos=0;
	   float Result=0.0f;

	   for (int i=0;i<FilterSize;i++)
	   {

		    Pos = (i-HalfFilter+globalY) * Width  + globalX ;

			if (globalY+(i-HalfFilter) >= 0 && globalY+(i-HalfFilter) < Height)
			{
				Value =	((float)MemSrc[Pos])*FilterCache[i];
	  		    Result=Result+Value;
			}
			else
			{
			    Value = ((float)MemSrc[GlobalOffset])*FilterCache[i];
			    Result=Result+Value;
			}
	   }
	   MemDst[GlobalOffset]=__float2half_rn(Result);
	 }
}
//==========================================================================
__global__ void FirstOrderStructureTensor_Kernel(unsigned short * MemSrc,unsigned short * MemIx2,unsigned short * MemIy2, unsigned short * MemIxIy,int Width, int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    float Ix,Iy;
    //------------------------------------------------------------------
	if (globalX>0 && globalX<(Width-1) && globalY>0 && globalY<(Height-1))
	{
		// Ori
		Ix=((float)(__half2float(MemSrc[GlobalOffset+1])-__half2float(MemSrc[GlobalOffset-1])))/2.0f;
		Iy=((float)(__half2float(MemSrc[GlobalOffset+Width])-__half2float(MemSrc[GlobalOffset-Width])))/2.0f;

	    MemIx2[GlobalOffset]= __float2half_rn(Ix*Ix);
	    MemIy2[GlobalOffset]= __float2half_rn(Iy*Iy);
	    MemIxIy[GlobalOffset]= __float2half_rn(Ix*Iy);
    }
	else
	{
		if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
		{
			MemIx2[GlobalOffset]= __float2half_rn(0.0f);
			MemIy2[GlobalOffset]= __float2half_rn(0.0f);
			MemIxIy[GlobalOffset]= __float2half_rn(0.0f);
		}
	}
}
//--------------------------------------------------------------------------
__global__ void CensusDerivates_Kernel(unsigned int * MemSrc1,unsigned int * MemSrc2,unsigned short * MemIx,unsigned short * MemIy, unsigned short * MemIt,int Width, int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    float Ix,Iy,It;
    unsigned int CentralPixel;
    //------------------------------------------------------------------
	if (globalX>1 && globalX<(Width-2) && globalY>1 && globalY<(Height-2))
	{
		CentralPixel=MemSrc1[GlobalOffset];

		//Ori
		/*Ix=(float)(HammingDistance(MemSrc2[GlobalOffset+1],CentralPixel))-(float)(HammingDistance(MemSrc2[GlobalOffset-1],CentralPixel));
		Iy=(float)(HammingDistance(MemSrc2[GlobalOffset+Width],CentralPixel))-(float)(HammingDistance(MemSrc2[GlobalOffset-Width],CentralPixel));
		It=(float)(HammingDistance(MemSrc2[GlobalOffset],CentralPixel))*(1.0f/4.0f);
*/
		Ix=((float)(HammingDistance2(MemSrc2[GlobalOffset+1],CentralPixel))-(float)(HammingDistance2(MemSrc2[GlobalOffset-1],CentralPixel)))*(1.0f/1.0f);
		Iy=((float)(HammingDistance2(MemSrc2[GlobalOffset+Width],CentralPixel))-(float)(HammingDistance2(MemSrc2[GlobalOffset-Width],CentralPixel)))*(1.0f/1.0f);
		It=((float)(HammingDistance2(MemSrc2[GlobalOffset],CentralPixel)))*(1.0f/4.0f);

	    MemIx[GlobalOffset]= __float2half_rn(Ix);
	    MemIy[GlobalOffset]= __float2half_rn(Iy);
	    MemIt[GlobalOffset]= __float2half_rn(It);
    }
	else
	{
		if (globalX>=0 && globalX<Width && globalY>=0 && globalY<Height)
		{
			MemIx[GlobalOffset]= __float2half_rn(0.0f);
			MemIy[GlobalOffset]= __float2half_rn(0.0f);
			MemIt[GlobalOffset]= __float2half_rn(0.0f);
		}
	}
}
//--------------------------------------------------------------------------

// End Kernels

//==========================================================================
//--------------------------------------------------------------------------
//==========================================================================
// Class Methods
//==========================================================================

//--------------------------------------------------------------------------
TCVFilters::TCVFilters(void * d_Gpu)
{
    Gpu = d_Gpu;

    //----------------------------------------------------------
    // Sigma=0.5
    //----------------------------------------------------------
    float Vector_0_5[3]= { -0.106506978919201f, -0.786986042161598f, -0.106506978919201f };
    float *p_Vector=Vector_0_5;
    d_FilterGauss_0_5= new TGpuMem::TGpuMemFloat(Gpu,3,1,1);
    d_FilterGauss_0_5->CopyToDevice(p_Vector);
    //----------------------------------------------------------
    // Sigma=1
    //----------------------------------------------------------
    float Vector_1 [3] =  { -0.27407f,-0.45186f, -0.27407f };
    //float Vector_1[7] { -0.0175595134796699f, -0.129748230171210f, -0.352692256349121f, 0.0f, -0.352692256349121f, -0.129748230171210f, -0.0175595134796699f };
    p_Vector=Vector_1;
    d_FilterGauss_1= new TGpuMem::TGpuMemFloat(Gpu,3,1,1);
    d_FilterGauss_1->CopyToDevice(p_Vector);
    //----------------------------------------------------------
    // Sigma=1.5
    //----------------------------------------------------------
    float Vector_1_5 [5] =  { -0.12008f,-0.23388f, -0.29208f, -0.23388f,-0.12008f};
    p_Vector=Vector_1_5;
    d_FilterGauss_1_5= new TGpuMem::TGpuMemFloat(Gpu,5,1,1);
    d_FilterGauss_1_5->CopyToDevice(p_Vector);
    //----------------------------------------------------------
    // Sigma=2
    //----------------------------------------------------------
    float Vector_2 [5] = { -0.15247f, -0.22184f, -0.25138f, -0.22184f, -0.15247f };
    p_Vector=Vector_2;
    d_FilterGauss_2= new TGpuMem::TGpuMemFloat(Gpu,5,1,1);
    d_FilterGauss_2->CopyToDevice(p_Vector);
    //----------------------------------------------------------
    // Sigma=2,5
    //----------------------------------------------------------
    float Vector_2_5 [7] = { -0.0923f, -0.1377f, -0.1751f, -0.1897f, -0.1751f, -0.1377f, -0.0923f };
    p_Vector=Vector_2_5;
    d_FilterGauss_2_5= new TGpuMem::TGpuMemFloat(Gpu,7,1,1);
    d_FilterGauss_2_5->CopyToDevice(p_Vector);
    //----------------------------------------------------------
    // Sigma=3
    //----------------------------------------------------------
    float Vector_3 [9] = { -0.0630f, -0.0929f, -0.1226f, -0.1449f, -0.1532f, -0.1449f, -0.1226f, -0.0929f, -0.0630f };
    p_Vector=Vector_3;
    d_FilterGauss_3= new TGpuMem::TGpuMemFloat(Gpu,9,1,1);
    d_FilterGauss_3->CopyToDevice(p_Vector);
    //----------------------------------------------------------
    // Sigma=3,5
    //----------------------------------------------------------
    float Vector_3_5 [11] =  { -0.046416f, -0.067019f, -0.089183f, -0.109374f, -0.123622f, -0.128772f, -0.123622f, -0.109374f, -0.089183f, -0.067019f, -0.046416f };
    p_Vector=Vector_3_5;
    d_FilterGauss_3_5= new TGpuMem::TGpuMemFloat(Gpu,11,1,1);
    d_FilterGauss_3_5->CopyToDevice(p_Vector);
    //----------------------------------------------------------
    // Sigma=4
    //----------------------------------------------------------
    float Vector_4 [11] = { -0.054884f, -0.072709f, -0.090488f, -0.105791f, -0.116189f, -0.119877f, -0.116189f, -0.105791f, -0.090488f, -0.072709f, -0.054884f };
    p_Vector=Vector_4;
    d_FilterGauss_4= new TGpuMem::TGpuMemFloat(Gpu,11,1,1);
    d_FilterGauss_4->CopyToDevice(p_Vector);
    //----------------------------------------------------------
    // Derivate size 5
    //----------------------------------------------------------
    //Ori
    float Derivate_5 [5] =  { 0.083333333333333f, -0.666666666666667f, 0.0f, 0.666666666666667f, -0.083333333333333f };
    p_Vector=Derivate_5;
    d_FilterDerivate_5= new TGpuMem::TGpuMemFloat(Gpu,5,1,1);
    d_FilterDerivate_5->CopyToDevice(p_Vector);
    //----------------------------------------------------------
    // Derivate size 5
    //----------------------------------------------------------
    float Vector_5 [7] = {-0.0175595134796699f, -0.129748230171210f, -0.352692256349121f, 0.0f, -0.352692256349121f, -0.129748230171210f, -0.0175595134796699f};
    p_Vector=Vector_5;
    d_FilterGauss_x= new TGpuMem::TGpuMemFloat(Gpu,7,1,1);
    d_FilterGauss_x->CopyToDevice(p_Vector);


}
//--------------------------------------------------------------------------
void TCVFilters::Median3x3(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Median3x3_Kernel<<<numBlocks, numThreads,(((TGpu *)Gpu)->GetBlockX()+2)*(((TGpu *)Gpu)->GetBlockY()+2)*sizeof(float)>>>(MemSrc->GetMemory(), MemDst->GetMemory(), MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVFilters::SeparableConvolution_H(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst,TGpuMem::TGpuMemFloat * MemFilter)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Separable_Convolution_H_Kernel<<<numBlocks, numThreads,(MemFilter->Size())*sizeof(float)>>>(MemSrc->GetMemory(), MemDst->GetMemory(),MemFilter->GetMemory(), MemFilter->Size(),MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();

}
//--------------------------------------------------------------------------
void TCVFilters::SeparableConvolution_V(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst,TGpuMem::TGpuMemFloat * MemFilter)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Separable_Convolution_H_Kernel<<<numBlocks, numThreads,(MemFilter->Size())*sizeof(float)>>>(MemSrc->GetMemory(), MemDst->GetMemory(),MemFilter->GetMemory(), MemFilter->Size(),MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVFilters::SeparableConvolution_H(TGpuMem::TGpuMemUChar * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst,TGpuMem::TGpuMemFloat * MemFilter)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Separable_Convolution_V_Kernel<<<numBlocks, numThreads,(MemFilter->Size())*sizeof(float)>>>(MemSrc->GetMemory(), MemDst->GetMemory(),MemFilter->GetMemory(), MemFilter->Size(),MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVFilters::SeparableConvolution_V(TGpuMem::TGpuMemUChar * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst,TGpuMem::TGpuMemFloat * MemFilter)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Separable_Convolution_V_Kernel<<<numBlocks, numThreads,(MemFilter->Size())*sizeof(float)>>>(MemSrc->GetMemory(), MemDst->GetMemory(),MemFilter->GetMemory(), MemFilter->Size(),MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVFilters::SeparableConvolution(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst,TGpuMem::TGpuMemFloat * MemFilter)
{
    TGpuMem::TGpuMemHalfFloat * MemAux= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1);

	SeparableConvolution_H(MemSrc,MemAux,MemFilter);
	SeparableConvolution_V(MemAux,MemDst,MemFilter);

	delete MemAux;
}
//--------------------------------------------------------------------------
void TCVFilters::FirstOrderStructureTensor(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemIx2,TGpuMem::TGpuMemHalfFloat * MemIy2, TGpuMem::TGpuMemHalfFloat * MemIxIy,TGpuMem::TGpuMemFloat * MemFilter)
{
    TGpuMem::TGpuMemHalfFloat * MemAux= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1);
    TGpuMem::TGpuMemHalfFloat * MemIx2Aux= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1);
    TGpuMem::TGpuMemHalfFloat * MemIy2Aux= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1);
    TGpuMem::TGpuMemHalfFloat * MemIxIyAux= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1);

    SeparableConvolution(MemSrc,MemAux,MemFilter);
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    FirstOrderStructureTensor_Kernel<<<numBlocks, numThreads>>>(MemAux->GetMemory(), MemIx2Aux->GetMemory(),MemIy2Aux->GetMemory(), MemIxIyAux->GetMemory(), MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();

    SeparableConvolution(MemIxIyAux,MemIxIy,MemFilter);
    SeparableConvolution(MemIx2Aux,MemIx2,MemFilter);
    SeparableConvolution(MemIy2Aux,MemIy2,MemFilter);
}
//--------------------------------------------------------------------------
void TCVFilters::CensusDerivates(TGpuMem::TGpuMemUInt * MemCensus1,TGpuMem::TGpuMemUInt * MemCensus2, TGpuMem::TGpuMemHalfFloat * MemIx,TGpuMem::TGpuMemHalfFloat * MemIy, TGpuMem::TGpuMemHalfFloat * MemIt)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemCensus1->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemCensus1->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    CensusDerivates_Kernel<<<numBlocks, numThreads>>>(MemCensus1->GetMemory(), MemCensus2->GetMemory(), MemIx->GetMemory(),MemIy->GetMemory(), MemIt->GetMemory(), MemCensus1->Width(), MemCensus1->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
TCVFilters::~TCVFilters()
{
    delete d_FilterGauss_0_5;
    delete d_FilterGauss_1;
    delete d_FilterGauss_2;
    delete d_FilterGauss_2_5;
    delete d_FilterGauss_3;
    delete d_FilterGauss_3_5;
    delete d_FilterGauss_4;
    delete d_FilterDerivate_5;
}
//--------------------------------------------------------------------------

