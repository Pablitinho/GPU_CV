#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include "TCVGeometry.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
//==========================================================================
// Kernels
//==========================================================================
__global__ void Resize_HF_Kernel(half * MemSrc, half * MemDst,int WidthSrc,int HeightSrc,int WidthDst,int HeightDst)
{
   //===============================================================================================
   //
   //===============================================================================================
   int globalX = blockIdx.x * blockDim.x + threadIdx.x;
   int globalY = blockIdx.y * blockDim.y + threadIdx.y;

   int OffsetMemDst = (globalY * WidthDst + globalX);

   float scaleWidth  =  (float)WidthDst / (float)(WidthSrc-1);
   float scaleHeight =  (float)HeightDst / (float)(HeightSrc-1);
   //===============================================================================================
   //
   //===============================================================================================
   if (globalX<WidthDst && globalY<HeightDst)
   {
        int x = (int)((float)globalX)/ scaleWidth;
        int y = (int)((float)globalY)/ scaleHeight;
        int x_1 = x-1;
        int y_1 = y-1;
        int x_2 = x+1;
        int y_2 = y+1;

        if(x_1 <= 0) x_1 = 0;
        if(y_1 <= 0) y_1 = 0;
        if(x_1 >= WidthSrc) x_1 = WidthSrc - 1;
        if(y_1 >= HeightSrc) y_1 = HeightSrc - 1;

        if(x_2 <= 0) x_2 = 0;
        if(y_2 <= 0) y_2 = 0;
        if(x_2 >= WidthSrc) x_2 = WidthSrc - 1;
        if(y_2 >= HeightSrc) y_2 = HeightSrc - 1;

		if(x <= 0) x = 0;
        if(y <= 0) y = 0;
        if(x >= WidthSrc) x = WidthSrc - 1;
        if(y >= HeightSrc) y = HeightSrc - 1;

        MemDst[OffsetMemDst] = __float2half( 0.25f*__half2float(MemSrc[y*WidthSrc+x]) + 0.125f*(__half2float(MemSrc[y*WidthSrc+x_1]) + __half2float(MemSrc[y*WidthSrc+x_2]) + __half2float(MemSrc[y_1*WidthSrc+x]) + __half2float(MemSrc[y_2*WidthSrc+x])) +
                          0.0625f*(__half2float(MemSrc[y_1*WidthSrc+x_1]) + __half2float(MemSrc[y_2*WidthSrc+x_1]) + __half2float(MemSrc[y_1*WidthSrc+x_2]) + __half2float(MemSrc[y_2*WidthSrc+x_2])));
   }
}
//--------------------------------------------------------------------------
__global__ void Resize_Kernel_Bilinear_HF(half* MemSrc, half * MemDst, int WidthSrc, int HeightSrc, int WidthDst, int HeightDst)
{
	//===============================================================================================
	//
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;

	int OffsetMemDst = (globalY * WidthDst + globalX);
	//===============================================================================================
	//
	//===============================================================================================
	if (globalX < WidthDst && globalY < HeightDst)
	{
		float x_ratio = ((float)(WidthSrc - 1)) / (float)(WidthDst);
		float y_ratio = ((float)(HeightSrc - 1)) / (float)(HeightDst);

		int x = (int)(x_ratio * globalX);
		int y = (int)(y_ratio * globalY);

		float x_diff = (x_ratio * globalX) - x;
		float y_diff = (y_ratio * globalY) - y;
		int index = y * WidthSrc + x;

		// range is 0 to 255 thus bitwise AND with 0xff
		float A = __half2float(MemSrc[index]);
		float B = __half2float(MemSrc[index + 1]);
		float C = __half2float(MemSrc[index + WidthSrc]);
		float D = __half2float(MemSrc[index + WidthSrc + 1]);

		// Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
		half Result = __float2half(A*(1.0f - x_diff)*(1.0f - y_diff) + B * (x_diff)*(1.0f - y_diff) +
					  C * (y_diff)*(1.0f - x_diff) + D * (x_diff*y_diff));

		MemDst[OffsetMemDst] = Result;
	}
}
template <typename T> __global__ void Resize_Kernel(T * MemSrc, T * MemDst, int WidthSrc, int HeightSrc, int WidthDst, int HeightDst)
{
	//===============================================================================================
	//
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;

	int OffsetMemDst = (globalY * WidthDst + globalX);

	float scaleWidth = (float)(WidthDst)/(float)(WidthSrc-1);
	float scaleHeight = (float)(HeightDst)/(float)(HeightSrc-1);
	
	//===============================================================================================
	//
	//===============================================================================================
	if (globalX < WidthDst && globalY < HeightDst)
	{
		int x = (int)((float)globalX) / scaleWidth;
		int y = (int)((float)globalY) / scaleHeight;
		int x_1 = x - 1;
		int y_1 = y - 1;
		int x_2 = x + 1;
		int y_2 = y + 1;

		if (x_1 <= 0) x_1 = 0;
		if (y_1 <= 0) y_1 = 0;
		if (x_1 >= WidthSrc) x_1 = WidthSrc - 1;
		if (y_1 >= HeightSrc) y_1 = HeightSrc - 1;

		if (x_2 <= 0) x_2 = 0;
		if (y_2 <= 0) y_2 = 0;
		if (x_2 >= WidthSrc) x_2 = WidthSrc - 1;
		if (y_2 >= HeightSrc) y_2 = HeightSrc - 1;

		if (x >= WidthSrc)
		{
			printf("out X\n");
		}
		if (y >= HeightSrc)
		{
			printf("out Y\n");
		}

		if (x <= 0) x = 0;
		if (y <= 0) y = 0;
		if (x >= WidthSrc) x = WidthSrc - 1;
		if (y >= HeightSrc) y = HeightSrc - 1;


		T Result = (T)(0.25f*(MemSrc[y*WidthSrc + x]) + 0.125f*((MemSrc[y*WidthSrc + x_1]) + (MemSrc[y*WidthSrc + x_2]) + (MemSrc[y_1*WidthSrc + x]) + (MemSrc[y_2*WidthSrc + x])) +
			0.0625f*((MemSrc[y_1*WidthSrc + x_1]) + (MemSrc[y_2*WidthSrc + x_1]) + (MemSrc[y_1*WidthSrc + x_2]) + (MemSrc[y_2*WidthSrc + x_2])));

		if (Result > 255) Result = 255;

		MemDst[OffsetMemDst] = Result;
	}
}
//--------------------------------------------------------------------------
template <typename T>__global__ void Resize_Kernel_Bilinear(T* MemSrc, T * MemDst, int WidthSrc, int HeightSrc, int WidthDst, int HeightDst)
{
	//===============================================================================================
	//
	//===============================================================================================
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;

	int OffsetMemDst = (globalY * WidthDst + globalX);
	//===============================================================================================
	//
	//===============================================================================================
	if (globalX < WidthDst && globalY < HeightDst)
	{
		float x_ratio = ((float)(WidthSrc - 1)) / (float)(WidthDst);
		float y_ratio = ((float)(HeightSrc - 1)) / (float)(HeightDst);

		int x = (int)(x_ratio * globalX);
		int y = (int)(y_ratio * globalY);

		float x_diff = (x_ratio * globalX) - x;
		float y_diff = (y_ratio * globalY) - y;
		int index = y*WidthSrc + x;

		// range is 0 to 255 thus bitwise AND with 0xff
		float A = (float)MemSrc[index];
		float B = (float)MemSrc[index + 1];
		float C = (float)MemSrc[index + WidthSrc];
		float D = (float)MemSrc[index + WidthSrc + 1];

		// Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
		T Result = (T)(A*(1.0f - x_diff)*(1.0f - y_diff) + B*(x_diff)*(1.0f - y_diff) +
							 C*(y_diff)*(1.0f - x_diff) + D*(x_diff*y_diff));

		MemDst[OffsetMemDst] = Result;
	}
}
//--------------------------------------------------------------------------
__device__ int neumann_bc(int x, int nx)
{
    if(x < 0)
    {
        x = 0;
    }
    else if (x >= nx)
    {
        x = nx - 1;
    }
    return x;
}
//--------------------------------------------------------------------------
__global__ void Warping_HF_Kernel(half * MemSrc, half * MemDst,half * U,half * V,bool Inverted,int Width,int Height)
{
	   //===============================================================================================
	   //
	   //===============================================================================================
	   int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	   int globalY = blockIdx.y * blockDim.y + threadIdx.y;

	   int GlobalOffset = (globalY * Width + globalX);
	   //===============================================================================================
	   //
	   //===============================================================================================
	   if (globalX<Width && globalY<Height)
	   {
		   float Result=0;
		   float uu=0,vv=0;
		   if (!Inverted)
		   {
		       uu =  (float)globalX-__half2float(U[GlobalOffset]);
			   vv =  (float)globalY-__half2float(V[GlobalOffset]);
		   }
		   else
		   {
			   uu =  (float)globalX+__half2float(U[GlobalOffset]);
			   vv =  (float)globalY+__half2float(V[GlobalOffset]);
		   }
		   //------------------------------------------
	       int sx = (uu < 0)? -1: 1;
	       int sy = (vv < 0)? -1: 1;

	       if (globalY==0) sy=1;

	       int x, y, dx, dy;

		   x  = neumann_bc((int) uu, Width);
	       y  = neumann_bc((int) vv, Height);
	       dx = neumann_bc((int) uu + sx, Width);
	       dy = neumann_bc((int) vv + sy, Height);

		   if (x>=0 && y>=0 && x<(Width) && y <(Height) && dx>=0 && dy >=0 && dx<(Width) && dy <(Height))
		   {
			  float p1 = __half2float(MemSrc[x  + Width * y]);
		      float p2 = __half2float(MemSrc[dx + Width * y]);
		      float p3 = __half2float(MemSrc[x  + Width * dy]);
			  float p4 = __half2float(MemSrc[dx + Width * dy]);

			  float e1 = ((float) sx * (uu - x));
			  float E1 = ((float) 1.0 - e1);
			  float e2 = ((float) sy * (vv - y));
			  float E2 = ((float) 1.0 - e2);

			 float w1 = E1 * p1 + e1 * p2;
		     float w2 = E1 * p3 + e1 * p4;

			 Result = E2 * w1 + e2 * w2;
		   }
		   //else Result = 0.0f;

		   MemDst[GlobalOffset]= __float2half(Result);
	   }
	   /*else
	   {
		   if (globalX>=0 && globalX<Width && globalY>=0 && globalY<Height)
			   MemDst[GlobalOffset]=__float2half_rn(0);
	   }*/
}
//--------------------------------------------------------------------------
__global__ void Warping_Uchar_Kernel(unsigned char * MemSrc, unsigned char * MemDst,half * U,half * V,bool Inverted,int Width,int Height)
{
	   //===============================================================================================
	   //
	   //===============================================================================================
	   int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	   int globalY = blockIdx.y * blockDim.y + threadIdx.y;

	   int GlobalOffset = (globalY * Width + globalX);
	   //===============================================================================================
	   //
	   //===============================================================================================
	   if (globalX>=0 && globalX<Width && globalY>=0 && globalY<Height)
	   {
			  float Result;
	          //float uu = (float) (globalPosX + U[iGlobalOffset]);
	          //float vv = (float) (globalPosY + V[iGlobalOffset]);
			  float uu,vv;
			  if (Inverted==0)
			  {
				   uu = (float)globalX-__half2float(U[GlobalOffset]);
				   vv =  (float)globalY-__half2float(V[GlobalOffset]);
			  }
			  else
			  {
				   uu =  (float)globalX+__half2float(U[GlobalOffset]);
				   vv =  (float)globalY+__half2float(V[GlobalOffset]);
			  }

	          int sx = (uu < 0)? -1: 1;
	          int sy = (vv < 0)? -1: 1;

	          //printf("[%d]: shared value is %d\n", threadIdx.x, sx);
	          int x, y, dx, dy;

			  x  = neumann_bc((int) uu, Width);
	          y  = neumann_bc((int) vv, Height);
	          dx = neumann_bc((int) uu + sx, Width);
	          dy = neumann_bc((int) vv + sy, Height);

	          //dx = neumann_bc((int) ceil(uu), Width);
	          //dy = neumann_bc((int) ceil(vv), Height);

			  if (x>=0 && y>=0 && x<(Width) && y <(Height) && dx>=0 && dy >=0 && dx<(Width) && dy <(Height))
			  //{
			    //if ((x+Width*y)< TotalSize && (dx + Width * y)<TotalSize && (x  + Width * dy)<TotalSize && (dx + Width * dy)<TotalSize)
				{
				  float p1 =  (MemSrc[x  + Width * y]);
				  float p2 =  (MemSrc[dx + Width * y]);
				  float p3 =  (MemSrc[x  + Width * dy]);
				  float p4 =  (MemSrc[dx + Width * dy]);

				  float e1 = ((float) sx * (uu - x));
				  float E1 = ((float) 1.0 - e1);
				  float e2 = ((float) sy * (vv - y));
				  float E2 = ((float) 1.0 - e2);

				  float w1 = E1 * p1 + e1 * p2;
				  float w2 = E1 * p3 + e1 * p4;

				  Result = E2 * w1 + e2 * w2;
			  }
			  else Result = 0.0f;

			  if (Result>255) Result=255;

			  MemDst[GlobalOffset]=  (unsigned char)(Result);
	   }
}
//==========================================================================
// End Kernels
//==========================================================================
//--------------------------------------------------------------------------
TCVGeometry::TCVGeometry(void * d_Gpu)
{
    Gpu = d_Gpu;
}
//--------------------------------------------------------------------------
void TCVGeometry::Resize(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemDst->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemDst->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
	Resize_HF_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(), MemDst->GetMemory(), MemSrc->Width(), MemSrc->Height(), MemDst->Width(), MemDst->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVGeometry::ResizeBilinear(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
	dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
	dim3 numBlocks = dim3(((TGpu *)Gpu)->iDivUp(MemDst->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemDst->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
	Resize_Kernel_Bilinear_HF << <numBlocks, numThreads >> > (MemSrc->GetMemory(), MemDst->GetMemory(), MemSrc->Width(), MemSrc->Height(), MemDst->Width(), MemDst->Height());
	cudaThreadSynchronize();
}
//--------------------------------------------------------------------------


void TCVGeometry::Resize(TGpuMem::TGpuMemUChar * MemSrc, TGpuMem::TGpuMemUChar * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemDst->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemDst->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Resize_Kernel<unsigned char><<<numBlocks, numThreads>>>(MemSrc->GetMemory(), MemDst->GetMemory(), MemSrc->Width(), MemSrc->Height(), MemDst->Width(), MemDst->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVGeometry::ResizeBilinear(TGpuMem::TGpuMemUChar * MemSrc, TGpuMem::TGpuMemUChar * MemDst)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
	dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
	dim3 numBlocks = dim3(((TGpu *)Gpu)->iDivUp(MemDst->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemDst->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
	Resize_Kernel_Bilinear<unsigned char> << <numBlocks, numThreads >> > (MemSrc->GetMemory(), MemDst->GetMemory(), MemSrc->Width(), MemSrc->Height(), MemDst->Width(), MemDst->Height());
	cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVGeometry::Warping(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst, TGpuMem::TGpuMemHalfFloat * MemU, TGpuMem::TGpuMemHalfFloat * MemV,bool Inverted)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemSrc->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemSrc->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Warping_HF_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(), MemDst->GetMemory(), MemU->GetMemory(), MemV->GetMemory(), Inverted, MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVGeometry::Warping(TGpuMem::TGpuMemUChar * MemSrc, TGpuMem::TGpuMemUChar * MemDst, TGpuMem::TGpuMemHalfFloat * MemU, TGpuMem::TGpuMemHalfFloat * MemV,bool Inverted)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemDst->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemDst->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Warping_Uchar_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(), MemDst->GetMemory(), MemU->GetMemory(), MemV->GetMemory(), Inverted, MemSrc->Width(), MemSrc->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
TCVGeometry::~TCVGeometry()
{
	//delete CV;
}
//--------------------------------------------------------------------------

