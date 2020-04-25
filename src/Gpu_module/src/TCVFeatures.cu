/*
 * TCVFeatures.cpp
 *
 *  Created on: 26/02/2015
 *      Author: pablo
 */

#include "TCVFeatures.h"
//#include <fstream>
//#include <iostream>
#include <stdio.h>
#include <typeinfo>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "CVCudaUtils.cuh"
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include "device_functions.h"
using namespace std;
//==========================================================================
// Kernels
//==========================================================================
__device__ void eig2x2(float a,float b,float c,float d,float *L1,float *L2,float *v11,float *v12,float *v21,float *v22)
{
	float T,D,T2;
	float tmp;
	//-------------------------------------
	T=a+d;
	D=a*d-b*c;
	//------------------------------------
	//if((4.0f-D)>0)
	//   tmp=sqrt(((T*T)/4.0)-D);
	//else
    tmp=sqrt(((T*T)/4.0f)-D);
	//-------------------------------------
    T2=(T/2.0f);
	//-------------------------------------
    *L1=T2-tmp;
    *L2=T2+tmp;
	//-------------------------------------
	//AutoVectores sin normalizar
	//-------------------------------------
	if (c!=0)
	{
		*v11=*L1-d;
		*v21=c;
		*v12=*L2-d;
		*v22=c;
	}
	else
	{
		*v11=b;
		*v21=*L1-a;
		*v12=b;
		*v22=*L2-a;
	}
	//-------------------------------------
    //Normalizamos los autovectores
	//-------------------------------------
    tmp= sqrt((*v11)*(*v11)+(*v21)*(*v21));
	if(tmp!=0)
	{
		*v11=  (*v11/tmp);
		*v21=  (*v21/tmp);
	}
	else
	{
		*v11=0.0f;
		*v21=0.0f;
	}
	//-------------------------------------
    tmp= sqrt((*v12)*(*v12)+(*v22)*(*v22));
    if(tmp!=0)
    {
		*v12=  (*v12/tmp);
		*v22=  (*v22/tmp);
	}
	else
	{
		*v12=0.0f;
		*v22=0.0f;
	}
	//-------------------------------------
	if (c==0 && b==0 && a==0 && d!=0)
	{
	    *v11=1.0;
		*v21=0.0;
		*v12=0.0;
		*v22=1.0;
	}
    //-------------------------------------
	if (c==0 && b==0 && a!=0 && d==0)
	{
	    *v11=0.0;
		*v21=1.0;
		*v12=1.0;
		*v22=0.0;
	}
	//-------------------------------------
}
__global__ void Census_Kernel(unsigned char * MemSrc, unsigned int * MemDst, int eps, int Width, int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    float Value;
    float ValueCenter;
    unsigned int Census=0;
    float Diff = 0;

	//int threadX = threadIdx.x+3;
	//int threadY = threadIdx.y+3;
    //int blockDimX = blockDim.x+2*3;
    //int blockDimY = blockDim.y+2*3;

    //int OffsetLocal = (threadY * blockDimX + threadX);

    extern __shared__ unsigned char DataCache[];
    //FillCacheRadius(DataCache, MemSrc, 3, Width, Height);
    //------------------------------------------------------------------
	if (globalX>1 && globalX<(Width-2) && globalY>1 && globalY<(Height-2))
	{
	    ValueCenter=MemSrc[GlobalOffset];
		//ValueCenter=DataCache[OffsetLocal];

	    #pragma unroll
		for(int dy=-1;dy<=1;dy++)
		{
			#pragma unroll
			for(int dx=-1;dx<=1;dx++)
			{
				if (!(dx==0 && dy==0))
			    {
					Value=MemSrc[(globalY+dy) * Width + (globalX+dx)];
					//Value=DataCache[(threadY+dy) * blockDimX + (threadX+dx)];
		            //---------------------------------------------------------------------
					// Ternary
					//---------------------------------------------------------------------
					Diff = ValueCenter - Value;

					Census = Census << 2;

					if (abs(Diff)<=eps)
					{
						Census=Census+1;
				    }
					else if (Diff> eps)
					{
						Census=Census+2;
					}
			    }
		   }
	    }
		#pragma unroll
		for(int dy=-2;dy<=2;dy++)
		{
			#pragma unroll
			for(int dx=-2;dx<=2;dx++)
			{
				if (!(dx==0 && dy==0) && !(abs(dx)==1 || abs(dy)==1))
				{
				    Value=MemSrc[(globalY+dy) * Width + (globalX+dx)];
					//Value=DataCache[(threadY+dy) * blockDimX + (threadX+dx)];
		            //---------------------------------------------------------------------
					// Ternary
					//---------------------------------------------------------------------
					Diff = ValueCenter - Value;
					Census = Census << 2;

					if (abs(Diff)<=eps)
					{
				       Census=Census+1;
					}
					else if (Diff> eps)
					{
				       Census=Census+2;
					}
			   }
		   }
	    }
		MemDst[GlobalOffset] = (Census);
	}
	else
	{
		if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
	    	MemDst[GlobalOffset] = 0;
	}
}
//--------------------------------------------------------------------------
__global__ void Census_Kernel(half * MemSrc, unsigned int * MemDst, int eps, int Width, int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    float Value;
    float ValueCenter;
    unsigned int Census=0;
    float Diff = 0;

	//int threadX = threadIdx.x+3;
	//int threadY = threadIdx.y+3;
    //int blockDimX = blockDim.x+2*3;
    //int blockDimY = blockDim.y+2*3;

    //int OffsetLocal = (threadY * blockDimX + threadX);

    //extern __shared__ unsigned char DataCache[];
    //FillCacheRadius(DataCache, MemSrc, 3, Width, Height);
    //------------------------------------------------------------------
	if (globalX>1 && globalX<(Width-2) && globalY>1 && globalY<(Height-2))
	{
	    ValueCenter= __half2float(MemSrc[GlobalOffset]);
		//ValueCenter=DataCache[OffsetLocal];

	    #pragma unroll
		for(int dy=-1;dy<=1;dy++)
		{
			#pragma unroll
			for(int dx=-1;dx<=1;dx++)
			{
				if (!(dx==0 && dy==0))
			    {
					Value= __half2float(MemSrc[(globalY+dy) * Width + (globalX+dx)]);
					//Value=DataCache[(threadY+dy) * blockDimX + (threadX+dx)];
		            //---------------------------------------------------------------------
					// Ternary
					//---------------------------------------------------------------------
					Diff = ValueCenter - Value;

					Census = Census << 2;

					if (abs(Diff)<=eps)
					{
						Census=Census+1;
				    }
					else if (Diff> eps)
					{
						Census=Census+2;
					}
			    }
		   }
	    }
		#pragma unroll
		for(int dy=-2;dy<=2;dy++)
		{
			#pragma unroll
			for(int dx=-2;dx<=2;dx++)
			{
				if (!(dx==0 && dy==0) && !(abs(dx)==1 || abs(dy)==1)&& globalX+dx>=0 )
				{
				    Value=__half2float(MemSrc[(globalY+dy) * Width + (globalX+dx)]);
					//Value=DataCache[(threadY+dy) * blockDimX + (threadX+dx)];
		            //---------------------------------------------------------------------
					// Ternary
					//---------------------------------------------------------------------
					Diff = ValueCenter - Value;
					Census = Census << 2;

					if (abs(Diff)<=eps)
					{
				       Census=Census+1;
					}
					else if (Diff> eps)
					{
				       Census=Census+2;
					}
			   }
		   }
	    }
		MemDst[GlobalOffset] = (Census);
	}
	else
	{
		if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
	    	MemDst[GlobalOffset] = 0;
	}
}
//--------------------------------------------------------------------------
__global__ void DiffusionWeight_Kernel(unsigned char * MemSrc,half * MemDst,float alpha,float beta, int Width, int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);

	float Result=0.0000;
    //------------------------------------------------------------------
	if (globalX>0 && globalX<(Width-1) && globalY>0 && globalY<(Height-1))
	{
	   float dH= ((float)(MemSrc[GlobalOffset-1])-(float)(MemSrc[GlobalOffset+1]))/2.0f;
	   float dV= ((float)(MemSrc[GlobalOffset+Width])-(float)(MemSrc[GlobalOffset-Width]))/2.0f;

	   float Grad= sqrt(dH*dH + dV*dV);

	   Result=exp(-alpha*pow(Grad,beta));
	   MemDst[GlobalOffset]= __float2half(Result);
    }
	else
	{
		if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
	    	MemDst[GlobalOffset] = __float2half(0.0f);
	}

}
//--------------------------------------------------------------------------
__global__ void DiffusionWeight_Kernel(half * MemSrc,half * MemDst,float alpha,float beta, int Width, int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);

	float Result=0.0000;
    //------------------------------------------------------------------
	if (globalX>0 && globalX<(Width-1) && globalY>0 && globalY<(Height-1))
	{
	   float dH= ((__half2float(MemSrc[GlobalOffset-1]))-(__half2float(MemSrc[GlobalOffset+1])))/2.0f;
	   float dV= ((__half2float(MemSrc[GlobalOffset+Width]))-(__half2float(MemSrc[GlobalOffset-Width])))/2.0f;

	   float Grad= sqrt(dH*dH + dV*dV);

	   Result=exp(-alpha*pow(Grad,beta));
	   MemDst[GlobalOffset]= __float2half(Result);
    }
	else
	{
		if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
	    	MemDst[GlobalOffset] = __float2half(0.0f);
	}
}
//--------------------------------------------------------------------------
__global__ void Derivate_Kernel(unsigned char * MemSrc,half * MemIx,half * MemIy, int Width, int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>0 && globalX<(Width-1) && globalY>0 && globalY<(Height-1))
	{
	    MemIx[GlobalOffset]= __float2half((float)((MemSrc[GlobalOffset+1])-(float)(MemSrc[GlobalOffset-1]))/2.0f);
	    MemIy[GlobalOffset]= __float2half((float)((MemSrc[GlobalOffset+Width])-(float)(MemSrc[GlobalOffset-Width]))/2.0f);
    }
	else
	{
		if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
	   {
		   MemIx[GlobalOffset]= __float2half(0.0f);
		   MemIy[GlobalOffset]= __float2half(0.0f);
	   }
	}
}
//--------------------------------------------------------------------------
__global__ void Derivate_Kernel(half * MemSrc,half * MemIx,half * MemIy, int Width, int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>0 && globalX<(Width-1) && globalY>0 && globalY<(Height-1))
	{
		
		// Ori
	    MemIx[GlobalOffset]= __float2half((float)(__half2float(MemSrc[GlobalOffset+1])-__half2float(MemSrc[GlobalOffset-1]))/2.0f);
	    MemIy[GlobalOffset]= __float2half((float)(__half2float(MemSrc[GlobalOffset+Width])-__half2float(MemSrc[GlobalOffset-Width]))/2.0f);
    }
	else
	{
	   if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
	   {
		   MemIx[GlobalOffset]= __float2half(0.0f);
		   MemIy[GlobalOffset]= __float2half(0.0f);
	   }
	}
}
//--------------------------------------------------------------------------
__global__ void Derivate_Kernel(unsigned char * MemSrc1,unsigned char * MemSrc2,half * MemIx,half * MemIy,half * MemIt, int Width, int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>0 && globalX<(Width-1) && globalY>0 && globalY<(Height-1))
	{
	    MemIx[GlobalOffset]= __float2half((float)((MemSrc1[GlobalOffset+1])-(float)(MemSrc1[GlobalOffset-1]))/2.0f);
	    MemIy[GlobalOffset]= __float2half((float)((MemSrc1[GlobalOffset+Width])-(float)(MemSrc1[GlobalOffset-Width]))/2.0f);
	    MemIt[GlobalOffset]= __float2half((float)((MemSrc1[GlobalOffset+Width])-(MemSrc2[GlobalOffset-Width])));
    }
	else
	{
		if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
	   {
		   MemIx[GlobalOffset]= __float2half(0.0f);
		   MemIy[GlobalOffset]= __float2half(0.0f);
		   MemIt[GlobalOffset]= __float2half(0.0f);
	   }
	}
}
//--------------------------------------------------------------------------
__global__ void Derivate_Kernel(half * MemSrc1,half * MemSrc2,half * MemIx,half * MemIy,half * MemIt, int Width, int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>0 && globalX<(Width-1) && globalY>0 && globalY<(Height-1))
	{
	    MemIx[GlobalOffset]= __float2half((float)(__half2float(MemSrc1[GlobalOffset+1])-(float)__half2float(MemSrc1[GlobalOffset-1]))/2.0f);
	    MemIy[GlobalOffset]= __float2half((float)(__half2float(MemSrc1[GlobalOffset+Width])-(float)__half2float(MemSrc1[GlobalOffset-Width]))/2.0f);
	    MemIt[GlobalOffset]= __float2half((float)(__half2float(MemSrc1[GlobalOffset+Width])-__half2float(MemSrc2[GlobalOffset-Width])));
    }
	else
	{
	   if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
	   {
		   MemIx[GlobalOffset]= __float2half(0.0f);
		   MemIy[GlobalOffset]= __float2half(0.0f);
		   MemIt[GlobalOffset]= __float2half(0.0f);
	   }
	}
}
//--------------------------------------------------------------------------
__global__ void EigenVectors_Kernel(half * MemIx2,half * MemIy2,half * MemIxIy,half * MemNx,half * MemNy ,int Width, int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int GlobalOffset = (globalY * Width + globalX);
    float Ix2,Iy2,IxIy;
    float a,b,c,d,L1,L2,v11,v12,v21,v22;
    //float Angle;
    //------------------------------------------------------------------
	if (globalX>0 && globalX<(Width-1) && globalY>0 && globalY<(Height-1))
	{
		Ix2=__half2float(MemIx2[GlobalOffset]);
		Iy2=__half2float(MemIy2[GlobalOffset]);
		IxIy=__half2float(MemIxIy[GlobalOffset]);

	    a = Ix2;
	    b = IxIy;
		c=b;
		d = Iy2;
	    //-----------------------------------------------------
		eig2x2(a,b,c,d, &L1, &L2, &v11, &v12, &v21, &v22);
		//-----------------------------------------------------
		if (L1>L2)
		{
		    if (isnan(v11) || isnan(v21) || isinf(v11) || isinf(v21))
			{
		        v11=0;
			    v21=0;
			}
		    MemNx[GlobalOffset]=__float2half(v11);
		    MemNy[GlobalOffset]=__float2half(v21);
		}
		else
	    {
		    if (isnan(v12) || isnan(v22) || isinf(v12) || isinf(v22))
			{
			    v12=0;
			    v22=0;
			}
		    MemNx[GlobalOffset]=__float2half(v12);
		    MemNy[GlobalOffset]=__float2half(v22);
		}
    }
	else
	{
		if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
		{
		    MemNx[GlobalOffset]= __float2half(0.0f);
		    MemNy[GlobalOffset]= __float2half(0.0f);
		}
	}
}
//==========================================================================
// End Kernels
//==========================================================================
TCVFeatures::TCVFeatures(void * d_Gpu)
{
	Gpu = d_Gpu;
}
//--------------------------------------------------------------------------
void TCVFeatures::DiffusionWeight(TGpuMem::TGpuMemUChar  * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst, float alpha,float beta)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    DiffusionWeight_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(),MemDst->GetMemory(),alpha, beta, MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVFeatures::DiffusionWeight(TGpuMem::TGpuMemHalfFloat  * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst, float alpha,float beta)
{
	TGpuMem::TGpuMemHalfFloat *MemGauss=new TGpuMem::TGpuMemHalfFloat(Gpu,(uint)MemSrc->Width(),(uint)MemSrc->Height(),1,false);
	((TGpu *)Gpu)->CV->Filters->SeparableConvolution(MemSrc,MemGauss,((TGpu *)Gpu)->CV->Filters->d_FilterGauss_1);

	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    DiffusionWeight_Kernel<<<numBlocks, numThreads>>>(MemGauss->GetMemory(),MemDst->GetMemory(),alpha, beta, MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();

    delete MemGauss;
}
//--------------------------------------------------------------------------
void TCVFeatures::Census(TGpuMem::TGpuMemUChar  * MemSrc, TGpuMem::TGpuMemUInt * MemDst,int eps)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Census_Kernel<<<numBlocks, numThreads,(((TGpu *)Gpu)->GetBlockX()+2)*(((TGpu *)Gpu)->GetBlockY()+2)*sizeof(unsigned char)>>>(MemSrc->GetMemory(),MemDst->GetMemory(), eps, MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVFeatures::Census(TGpuMem::TGpuMemHalfFloat  * MemSrc, TGpuMem::TGpuMemUInt * MemDst,int eps)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Census_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(),MemDst->GetMemory(), eps, MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVFeatures::Derivates(TGpuMem::TGpuMemUChar  * MemSrc, TGpuMem::TGpuMemHalfFloat * MemIx,TGpuMem::TGpuMemHalfFloat * MemIy)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Derivate_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(),MemIx->GetMemory(),MemIy->GetMemory(), MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVFeatures::Derivates(TGpuMem::TGpuMemHalfFloat  * MemSrc, TGpuMem::TGpuMemHalfFloat * MemIx,TGpuMem::TGpuMemHalfFloat * MemIy)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Derivate_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(),MemIx->GetMemory(),MemIy->GetMemory(), MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVFeatures::Derivates(TGpuMem::TGpuMemUChar * MemSrc1,TGpuMem::TGpuMemUChar * MemSrc2, TGpuMem::TGpuMemHalfFloat * MemIx,TGpuMem::TGpuMemHalfFloat * MemIy,TGpuMem::TGpuMemHalfFloat * MemIt)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc1->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc1->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Derivate_Kernel<<<numBlocks, numThreads>>>(MemSrc1->GetMemory(),MemSrc2->GetMemory(),MemIx->GetMemory(),MemIy->GetMemory(),MemIt->GetMemory(), MemSrc1->Width(), MemSrc1->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVFeatures::Derivates(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2, TGpuMem::TGpuMemHalfFloat * MemIx,TGpuMem::TGpuMemHalfFloat * MemIy,TGpuMem::TGpuMemHalfFloat * MemIt)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc1->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc1->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Derivate_Kernel<<<numBlocks, numThreads>>>(MemSrc1->GetMemory(),MemSrc2->GetMemory(),MemIx->GetMemory(),MemIy->GetMemory(),MemIt->GetMemory(), MemSrc1->Width(), MemSrc1->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVFeatures::EigenVectors(TGpuMem::TGpuMemUChar  * MemSrc, TGpuMem::TGpuMemHalfFloat * MemNx,TGpuMem::TGpuMemHalfFloat * MemNy,TGpuMem::TGpuMemFloat * MemFilter)
{
	//----------------------------------------------------------------------------------------------------
    TGpuMem::TGpuMemHalfFloat * MemIx= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1, false);
    TGpuMem::TGpuMemHalfFloat * MemIy= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1, false);
    TGpuMem::TGpuMemHalfFloat * MemIxAux= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1, false);
    TGpuMem::TGpuMemHalfFloat * MemIyAux= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1, false);
	//----------------------------------------------------------------------------------------------------
    Derivates(MemSrc,MemIx,MemIy);
    ((TGpu *)Gpu)->CV->Filters->SeparableConvolution(MemIx,MemIxAux,MemFilter);
    ((TGpu *)Gpu)->CV->Filters->SeparableConvolution(MemIy,MemIyAux,MemFilter);
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    //EigenVectors_UChar_Kernel<<<numBlocks, numThreads>>>(MemIxAux->GetMemory(),MemIyAux->GetMemory(),MemIx->GetMemory(),MemIy->GetMemory(), MemSrc->Width(), MemSrc->Height());
    //EigenVectors_Kernel<<<numBlocks, numThreads>>>(MemIxAux->GetMemory(),MemIyAux->GetMemory(),MemNx->GetMemory(),MemNy->GetMemory(), MemSrc->Width(), MemSrc->Height());
    //cudaThreadSynchronize();
    //((TGpu *)Gpu)->CV->Filters->SeparableConvolution(MemIx,MemNx,((TGpu *)Gpu)->CV->Filters->d_FilterGauss_2);
    //((TGpu *)Gpu)->CV->Filters->SeparableConvolution(MemIy,MemNy,((TGpu *)Gpu)->CV->Filters->d_FilterGauss_2);
	//----------------------------------------------------------------------------------------------------

	delete MemIx;
	delete MemIy;
	delete MemIxAux;
	delete MemIyAux;
}
//--------------------------------------------------------------------------
void TCVFeatures::EigenVectors(TGpuMem::TGpuMemHalfFloat  * MemSrc, TGpuMem::TGpuMemHalfFloat * MemNx,TGpuMem::TGpuMemHalfFloat * MemNy,TGpuMem::TGpuMemFloat * MemFilter)
{
	//----------------------------------------------------------------------------------------------------
    TGpuMem::TGpuMemHalfFloat * MemIx2= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1, false);
    TGpuMem::TGpuMemHalfFloat * MemIy2= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1, false);
    TGpuMem::TGpuMemHalfFloat * MemIxIy= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1, false);
    TGpuMem::TGpuMemHalfFloat * MemIxAux= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1, false);
    TGpuMem::TGpuMemHalfFloat * MemIyAux= new TGpuMem::TGpuMemHalfFloat(Gpu,MemSrc->Width(), MemSrc->Height(),1, false);
	//----------------------------------------------------------------------------------------------------
    ((TGpu *)Gpu)->CV->Filters->FirstOrderStructureTensor(MemSrc,MemIx2,MemIy2,MemIxIy,MemFilter);
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    EigenVectors_Kernel<<<numBlocks, numThreads>>>(MemIx2->GetMemory(),MemIy2->GetMemory(),MemIxIy->GetMemory(), MemNx->GetMemory(),MemNy->GetMemory(), MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
	//----------------------------------------------------------------------------------------------------

	delete MemIx2;
	delete MemIy2;
	delete MemIxIy;
	delete MemIxAux;
	delete MemIyAux;
}
//--------------------------------------------------------------------------
TCVFeatures::~TCVFeatures()
{

}
//--------------------------------------------------------------------------


