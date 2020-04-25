#ifndef CVCUDAUTILS_CUH_
#define CVCUDAUTILS_CUH_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "cuda_fp16.h"
//--------------------------------------------------------------------------
#define NUMPI 3.14159265358979323846
#define NUMPI_2 1.57079632679489661923
//--------------------------------------------------------------------------
/*inline __device__ void FillCache(volatile float *DataCache, half * MemSrc, int radius, int Width,  int Height)
{
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int OffsetIm = (globalY * Width + globalX);

	int threadX = threadIdx.x+1;
	int threadY = threadIdx.y+1;
	int blockDimX = blockDim.x+2;
	int blockDimY = blockDim.y+2;

    int OffsetLocal = (threadY * blockDimX + threadX);

    DataCache[OffsetLocal]=__half2float(MemSrc[OffsetIm]);

    //printf("[%d]: shared value is %d\n", threadIdx.x, DataCache[OffsetLocal]);

	if (threadX==1 || threadX==blockDimX-2 || threadY==1 || threadX==blockDimY-2)
	{
		// Left-up Corner
		if (threadX==1 && threadY==1)
		{
			if (globalX!=0)
				DataCache[0]=__half2float(MemSrc[OffsetIm-Width-1]);
			else DataCache[0]=255.0f;
		}
		// Left-down Corner
		if (threadX==1 && threadX==blockDimY-2)
		{
			if (globalX!=0 && globalY!=Height-1)
				DataCache[OffsetLocal-1+blockDimX]=__half2float(MemSrc[OffsetIm+Width-1]);
			else DataCache[0]=255.0f;
		}
		//Right-up Corner
		if (threadX==blockDimX-2 && threadY==1)
		{
			if (globalX!=Width-1 && globalY!=0)
				DataCache[OffsetLocal+1-blockDimX]=__half2float(MemSrc[OffsetIm-Width+1]);
			else DataCache[OffsetLocal+1-blockDimX]=255.0f;
		}
		//Right-down Corner
		if (threadX==blockDimX-2 && threadX==blockDimY-2)
		{
			if (globalX!=Width-1 && globalY!=Height-1)
				DataCache[OffsetLocal+1+blockDimX]=__half2float(MemSrc[OffsetIm+Width+1]);
			else DataCache[OffsetLocal+1+blockDimX]=255.0f;
		}

		//-----------------------------------------------------------------------------------------------
		// Left
		if (threadX==1)
		{
			 if (globalX!=0)
				 DataCache[OffsetLocal-1]=__half2float(MemSrc[OffsetIm-1]);
			 else DataCache[OffsetLocal-1]=0.0f;
		}
		//Right
		if (threadX==blockDimX-2)
		{
			 if (globalX!=Width-1)
				 DataCache[OffsetLocal+1]=__half2float(MemSrc[OffsetIm+1]);
			 else DataCache[OffsetLocal+1]=0.0f;
		}
		//-----------------------------------------------------------------------------------------------
		//Up
		if (threadY==1)
		{
			if (globalY!=0)
				DataCache[OffsetLocal-blockDimX]=__half2float(MemSrc[OffsetIm-Width]);
			else DataCache[OffsetLocal-blockDimX]=0.0f;
		}
		//-----------------------------------------------------------------------------------------------
		//Down
		if (threadX==blockDimY-2)
		{
			if (globalY!=Height-1)
				DataCache[(threadY) * blockDimX + blockDimX + (threadX)]=__half2float(MemSrc[OffsetIm+Width]);
			else DataCache[(threadY) * blockDimX + blockDimX + (threadX)]=0.0f;
		}
	}
	__syncthreads();
}
//--------------------------------------------------------------------------
inline __device__ void FillCacheRadius(volatile unsigned char *DataCache, unsigned char * MemSrc, int radius, int Width,  int Height)
{
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int OffsetIm = (globalY * Width + globalX);

	int threadX = threadIdx.x+radius;
	int threadY = threadIdx.y+radius;
	int blockDimX = blockDim.x+2*radius;
	int blockDimY = blockDim.y+2*radius;

    int OffsetLocal = (threadY * blockDimX + threadX);

    DataCache[OffsetLocal]=(MemSrc[OffsetIm]);
    unsigned char DefaultValue=0;
    //printf("[%d]: shared value is %d\n", threadIdx.x, DataCache[OffsetLocal]);

	if (threadX==radius || threadX==blockDimX-1-radius || threadY==radius || threadX==blockDimY-1-radius)
	{
		// Left-up Corner
		if (threadX==radius && threadY==radius)
		{
		    for(int i=0;i<radius;i++)
			    for(int j=0;j<radius;j++)
			        if (globalX-j>=0 && globalY-i>=0)
			       	    DataCache[OffsetLocal-blockDimX*i-j]=(MemSrc[OffsetIm-Width*i-j]);
				    else DataCache[OffsetLocal-blockDimX*i-j]=DefaultValue;
		}
		// Left-down Corner
		if (threadX==radius && threadX==blockDimY-1-radius)
		{


		    for(int i=0;i<radius;i++)
			    for(int j=0;j<radius;j++)
			        if (globalX-j>=0 && globalY-i>=0)
			       	    DataCache[OffsetLocal+blockDimX*i-j]=(MemSrc[OffsetIm+Width*i-j]);
				    else DataCache[OffsetLocal+blockDimX*i-j]=DefaultValue;
		}
		//Right-up Corner
		if (threadX==blockDimX-1-radius && threadY==radius)
		{

		    for(int i=0;i<radius;i++)
			    for(int j=0;j<radius;j++)
			        if (globalX-j>=0 && globalY-i>=0)
			       	    DataCache[OffsetLocal-blockDimX*i+j]=(MemSrc[OffsetIm-Width*i+j]);
				    else DataCache[OffsetLocal-blockDimX*i+j]=DefaultValue;

		}
		//Right-down Corner
		if (threadX==blockDimX-1-radius && threadX==blockDimY-1-radius)
		{

		    for(int i=0;i<radius;i++)
			    for(int j=0;j<radius;j++)
			        if (globalX-j>=0 && globalY-i>=0)
			       	    DataCache[OffsetLocal+blockDimX*i+j]=(MemSrc[OffsetIm+Width*i+j]);
				    else DataCache[OffsetLocal+blockDimX*i+j]=DefaultValue;
		}
		//-----------------------------------------------------------------------------------------------
		// Left
		if (threadX==radius)
		{
			 if (globalX!=0)
			 {
				 for(int i=0;i<radius;i++)
					 if (globalX-i>=0)
				         DataCache[OffsetLocal-i]=(MemSrc[OffsetIm-i]);
					 else DataCache[OffsetLocal+i]=DefaultValue;
			 }
			 else
			 {
				 for(int i=0;i<radius;i++)
				     DataCache[OffsetLocal-i]=DefaultValue;
			 }
		}
		//Right
		if (threadX==blockDimX-1-radius)
		{
			 if (globalX!=Width-1)
				 for(int i=0;i<radius;i++)
					 if (globalX+i>=0)
				         DataCache[OffsetLocal+i]=(MemSrc[OffsetIm+i]);
					 else DataCache[OffsetLocal+i]=DefaultValue;
			 else
			 {
				 for(int i=0;i<radius;i++)
				     DataCache[OffsetLocal+i]=DefaultValue;
			 }
		}
		//-----------------------------------------------------------------------------------------------
		//Up
		if (threadY==radius)
		{
			if (globalY!=0)
				 for(int i=0;i<radius;i++)
					 if (globalY-i>=0)
				         DataCache[OffsetLocal-blockDimX*i]=(MemSrc[OffsetIm-Width*i]);
					 else DataCache[OffsetLocal-blockDimX*i]=DefaultValue;
			 else
			 {
				 for(int i=0;i<radius;i++)
				     DataCache[OffsetLocal-i*blockDimX]=DefaultValue;
			 }
			//else DataCache[OffsetLocal-blockDimX]=DefaultValue;
		}
		//-----------------------------------------------------------------------------------------------
		//Down
		if (threadX==blockDimY-1-radius)
		{
			if (globalY!=Height-1)
				 for(int i=0;i<radius;i++)
					 if (globalY-i>=0)
						 DataCache[OffsetLocal+blockDimX*i]=(MemSrc[OffsetIm+Width*i]);

					 else DataCache[OffsetLocal+blockDimX*i]=DefaultValue;
			 else
			 {
				 for(int i=0;i<radius;i++)
				     DataCache[OffsetLocal+i*blockDimX]=DefaultValue;
			 }
			//else DataCache[(threadY) * blockDimX + blockDimX + (threadX)]=DefaultValue;
		}
	}
	__syncthreads();
}*/
//--------------------------------------------------------------------------
inline __device__ float Range(float valorIn, float maxIn, float minIn,float maxOut,float minOut){

       if (valorIn < minIn)
           valorIn = minIn;

       if (valorIn > maxIn)
           valorIn = maxIn;

       float V = valorIn / (maxIn - minIn) * (maxOut - minOut) + minOut;
       if (isnan(V) || isinf(V))
           return 0.0f;
       else
           return V;
}
//==========================================================================
/*inline __device__ void ReplicatePixels(half * Mem, float Value,int globalX,int globalY,int GlobalOffset,int Width,int Height)
{
	   half ValueHF=__float2half_rn(Value);
	   if (globalX==3 )
	   {
		   Mem[GlobalOffset-1]=ValueHF;
		   Mem[GlobalOffset-2]=ValueHF;
		   Mem[GlobalOffset-3]=ValueHF;
	   }
	   if (globalX==Width-4)
	   {
		   Mem[GlobalOffset+1]=ValueHF;
		   Mem[GlobalOffset+2]=ValueHF;
		   Mem[GlobalOffset+3]=ValueHF;
	   }
	   if (globalY==3)
	   {
		   Mem[GlobalOffset-Width]=ValueHF;
		   Mem[GlobalOffset-2*Width]=ValueHF;
		   Mem[GlobalOffset-3*Width]=ValueHF;
	   }
	   if (globalY==Height-4)
	   {
		   Mem[GlobalOffset+Width]=ValueHF;
		   Mem[GlobalOffset+2*Width]=ValueHF;
		   Mem[GlobalOffset+3*Width]=ValueHF;
	   }

}*/
//--------------------------------------------------------------------------
inline __device__ void ReplicatePixels(half * Mem, float Value,int globalX,int globalY,int GlobalOffset,int Width,int Height)
{
	   half ValueHF=__float2half(Value);
	   if (globalX==2 )
	   {
		   Mem[GlobalOffset-1]=ValueHF;
		   Mem[GlobalOffset-2]=ValueHF;
	   }
	   if (globalX==Width-3)
	   {
		   Mem[GlobalOffset+1]=ValueHF;
		   Mem[GlobalOffset+2]=ValueHF;
	   }
	   if (globalY==2)
	   {
		   Mem[GlobalOffset-Width]=ValueHF;
		   Mem[GlobalOffset-2*Width]=ValueHF;
	   }
	   if (globalY==Height-3)
	   {
		   Mem[GlobalOffset+Width]=ValueHF;
		   Mem[GlobalOffset+2*Width]=ValueHF;
	   }

}
//--------------------------------------------------------------------------
inline __device__ float f_sqrt(float x)
{
	/*
	float xhalf = 0.5f*x;
	int i = *(int *)&x;
	i = 0x5f3759df - (i >> 1);
	x = *(float *)&i;
	x = x * (1.5f - xhalf * x * x);
	return 1.0f/x;
	*/
	return 1.0f/rsqrt(x);
}
//--------------------------------------------------------------------------
/*inline __device__ void eig2x2(float a,float b,float c,float d,float *L1,float *L2,float *v11,float *v12,float *v21,float *v22)
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
}*/
//===============================================================================================
// Trigonometry & Math
//===============================================================================================
inline __device__ float deg2rad(float value)
{
    return (NUMPI*value/180.0f);
}
//===============================================================================================
inline __device__ float rad2deg(float value)
{
    return (180.0f*value/NUMPI);
}
//===============================================================================================
inline __device__ float Range_Value(float valorIn, float maxIn, float minIn,float maxOut,float minOut){

       if (valorIn < minIn)
           valorIn = minIn;

       if (valorIn > maxIn)
           valorIn = maxIn;

       float scale = (maxOut - minOut) / (maxIn - minIn);
       float V = (minOut + ((valorIn - minIn) * scale));

       if (isnan(V) || isinf(V))
           return 0.0f;
       else
           return V;

}
//===============================================================================================
inline __device__ float sign(float x)
{
	return (x/abs(x));
}
//===============================================================================================
/*__device__ inline uint HammingDistance(uint x,uint y)
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
}*/
//===============================================================================================

#endif
