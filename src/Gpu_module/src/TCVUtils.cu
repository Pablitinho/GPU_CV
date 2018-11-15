/*
 * TCVUtils.cpp
 *
 *  Created on: 04/03/2015
 *      Author: pablo
 */

#include "TCVUtils.h"
#include "defines.h"
//==========================================================================
// Kernels
//==========================================================================
__global__ void Padding_Kernel(unsigned short * MemSrc, unsigned short * MemDst,uint Xoffset,uint Yoffset, float Value,int WidthSrc, int HeightSrc,int WidthDst, int HeightDst)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    int iGlobalOffsetDst =(globalY+Yoffset) * WidthDst + (globalX+Xoffset);
    int iGlobalOffsetSrc =globalY * WidthSrc + globalX;
    //------------------------------------------------------------------
	if(globalX>=0 && globalY>=0 && globalX<WidthSrc && globalY<HeightSrc)
	{
		MemDst[iGlobalOffsetDst]= MemSrc[iGlobalOffsetSrc];
	}
	else if (globalX>=0 && globalY>=0 && globalX<WidthDst && globalY<HeightDst)
	{
		MemDst[iGlobalOffsetDst]= __float2half_rn(Value);
	}
}
//--------------------------------------------------------------------------
__global__ void Replicate_H_Kernel(unsigned short * MemU, unsigned short * MemV,unsigned short * MemUOut, unsigned short * MemVOut,int Width, int Height)
{
	//------------------------------------------------------------------
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    if (globalY==2) globalY=Height-1;
    if (globalY==3) globalY=Height-2;
    /*
    if (globalY==3) globalY=Height-1;
    if (globalY==4) globalY=Height-2;
    if (globalY==5) globalY=Height-3;
*/

    int GlobalOffset = (globalY * Width + globalX);
	//------------------------------------------------------------------

   if (globalX>=0 && globalX<Width && globalY>=0 && globalY<Height)
   {
	   if (globalX <=2)
	   {
	       MemUOut[GlobalOffset]=MemU[(globalY * Width + 3)];
		   MemVOut[GlobalOffset]=MemV[(globalY * Width + 3)];
	   }

	   if (globalX>=(Width-4))
	   {
		   MemUOut[GlobalOffset]=MemU[globalY * Width + (Width-4)];
		   MemVOut[GlobalOffset]=MemV[globalY * Width + (Width-4)];
	   }
	   if (globalY<=2)
	   {
		   MemUOut[GlobalOffset]=MemU[(3 * Width + globalX)];
		   MemVOut[GlobalOffset]=MemV[(3 * Width + globalX)];
	   }

	   if (globalY>=(Height-4))
	   {
		   MemUOut[GlobalOffset]=MemU[((Height-4) * Width + globalX)];
		   MemVOut[GlobalOffset]=MemV[((Height-4) * Width + globalX)];
	   }

   }
}
//==========================================================================
__global__ void Replicate_V_Kernel(unsigned short * MemU, unsigned short * MemV,unsigned short * MemUOut, unsigned short * MemVOut,int Width, int Height)
{
	//------------------------------------------------------------------
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    if (globalX==2) globalX=Width-1;
    if (globalX==3) globalX=Width-2;
    /*
    if (globalX==3) globalX=Width-1;
    if (globalX==4) globalX=Width-2;
    if (globalX==5) globalX=Width-3;
    */

    int GlobalOffset = (globalY * Width + globalX);
	//------------------------------------------------------------------

   if (globalX>=0 && globalX<Width && globalY>=0 && globalY<Height)
   {
	   if (globalX <=2)
	   {
	       MemUOut[GlobalOffset]=MemU[(globalY * Width + 3)];
		   MemVOut[GlobalOffset]=MemV[(globalY * Width + 3)];
	   }

	   if (globalX>=(Width-3))
	   {
		   MemUOut[GlobalOffset]=MemU[globalY * Width + (Width-4)];
		   MemVOut[GlobalOffset]=MemV[globalY * Width + (Width-4)];
	   }
	   if (globalY<=2)
	   {
		   MemUOut[GlobalOffset]=MemU[(3 * Width + globalX)];
		   MemVOut[GlobalOffset]=MemV[(3 * Width + globalX)];
	   }

	   if (globalY>=(Height-3))
	   {
		   MemUOut[GlobalOffset]=MemU[((Height-4) * Width + globalX)];
		   MemVOut[GlobalOffset]=MemV[((Height-4) * Width + globalX)];
	   }

   }
}
//==========================================================================
__global__ void OpticalFlowToColor_Kernel(unsigned short * MemU,unsigned short * MemV, unsigned char * MemDst,int * MemColorWheel,float maxRad,int ncols,int Width,int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = (blockIdx.x * blockDim.x + threadIdx.x)*1;
    int globalY = (blockIdx.y * blockDim.y + threadIdx.y)*1;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
    float fx,fy;
    //ncols=55;
   //===============================================================================================
   if (globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
   {
	    fx=__half2float(MemU[GlobalOffset]);
	    fy=__half2float(MemV[GlobalOffset]);

	    /*if (fx>maxRad) fx=0.99;
	    else fx=fx/maxRad;
	    if (fy>maxRad) fy=0.99;
	    else fy=fy/maxRad;*/

	    fx=fx/maxRad;
	    fy=fy/maxRad;

	    float rad = sqrt(fx * fx + fy * fy);
	    if (sqrt(fx*fx+fy*fy)>=1)
	    	rad=1;

	    float a = atan2(-fy, -fx) / M_PI;
	    float fk = (a + 1.0) / 2.0 * (ncols-1);
	    int k0 = (int)fk;
	    int k1 = (k0 + 1) % ncols;
	    float f = fk - k0;
	    //f = 0; // uncomment to see original color wheel
	    float col,col0,col1;
	    for (int b = 0; b < 3; b++)
	    {
			col0 = (float)MemColorWheel[k0*3+b] / 255.0;
			col1 = (float)MemColorWheel[k1*3+b] / 255.0;

		    col = (1 - f) * col0 + f * col1;
			if (rad <= 1)
				col = 1 - rad * (1 - col); // increase saturation with radius
			else
				col *= .75; // out of range
			//printf("ValorCol=%f Valorfx=%f Valorfy=%f\n",maxRad,fx,fy);
			//printf("NulCol=%d Valorfx=%f Valorfy=%f\n",ncols,fx,fy);

			MemDst[GlobalOffset*3+2-b]= (unsigned char)(255.0 * col);
	    }
   }
}
//==========================================================================
__global__ void StereoToColor_Kernel(unsigned short * MemU, unsigned char * MemDst,int * MemColorWheel,int ncols,int Width,int Height)
{
	//===============================================================================================
    //
	//===============================================================================================
	int globalX = (blockIdx.x * blockDim.x + threadIdx.x)*1;
    int globalY = (blockIdx.y * blockDim.y + threadIdx.y)*1;

    int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
    float fx;
    //ncols=55;
   //===============================================================================================
   if (globalX>=0 && globalY>=0 && globalX<Width && globalY<Height)
   {
	    fx=__half2float(MemU[GlobalOffset]);

	    float rad = abs(fx);

	    float a = atan2( 0.0f,-fx) / M_PI;
	    float fk = (a + 1.0) / 2.0 * (ncols-1);
	    int k0 = (int)fk;
	    int k1 = (k0 + 1) % ncols;
	    float f = fk - k0;
	    //f = 0; // uncomment to see original color wheel
	    float col,col0,col1;
	    for (int b = 0; b < 3; b++)
	    {
			col0 = (float)MemColorWheel[k0*3+b] / 255.0;
			col1 = (float)MemColorWheel[k1*3+b] / 255.0;

		    col = (1 - f) * col0 + f * col1;
			if (rad <= 1)
				col = 1 - rad * (1 - col); // increase saturation with radius
			else
				col *= .75; // out of range
			//printf("ValorCol=%f Valorfx=%f Valorfy=%f\n",maxRad,fx,fy);
			//printf("NulCol=%d Valorfx=%f Valorfy=%f\n",ncols,fx,fy);

			MemDst[GlobalOffset*3+2-b]= (unsigned char)(255.0 * col);
	    }
   }
}
//==========================================================================
//==========================================================================
// End Kernels
//==========================================================================
//--------------------------------------------------------------------------
TCVUtils::TCVUtils(void * d_Gpu)
{
	Gpu = d_Gpu;
	//-----------------------------------------------
	// OF and Stereo Color
	//-----------------------------------------------
    ncols=0;
	MemColorWheel = new TGpuMem::TGpuMemInt(Gpu,60,1,3);
	int * ptr_Color=Makecolorwheel();
	MemColorWheel->CopyToDevice(ptr_Color);
	delete ptr_Color;
	//-----------------------------------------------
}
//--------------------------------------------------------------------------
void TCVUtils::Padding(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst,uint Xoffset,uint Yoffset,float Value)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemDst->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemDst->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Padding_Kernel<<<numBlocks, numThreads>>>(MemSrc->GetMemory(),MemDst->GetMemory(), Xoffset,Yoffset,Value,MemSrc->Width(), MemSrc->Height(),MemDst->Width(), MemDst->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVUtils::ReplicateEdges(TGpuMem::TGpuMemHalfFloat * MemU, TGpuMem::TGpuMemHalfFloat * MemV)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemU->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(4, numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Replicate_H_Kernel<<<numBlocks, numThreads>>>(MemU->GetMemory(), MemV->GetMemory(),MemU->GetMemory(), MemV->GetMemory(), MemU->Width(), MemU->Height());
    cudaThreadSynchronize();
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(4, numThreads.x), ((TGpu *)Gpu)->iDivUp(MemU->Height(), numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Replicate_V_Kernel<<<numBlocks, numThreads>>>(MemU->GetMemory(), MemV->GetMemory(),MemU->GetMemory(), MemV->GetMemory(), MemU->Width(), MemU->Height());
    cudaThreadSynchronize();
}
//--------------------------------------------------------------------------
void TCVUtils::Setcols(int *colorwheel,int r, int g, int b, int k)
{
    colorwheel[k*3] = r;
    colorwheel[k*3+1] = g;
    colorwheel[k*3+2] = b;
}
//--------------------------------------------------------------------------
int * TCVUtils::Makecolorwheel()
{
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    ncols = RY + YG + GC + CB + BM + MR;
	int *colorwheel=new int[ncols*3];
    if (ncols > 60)
	exit(1);
    int i;
    int k = 0;
    for (i = 0; i < RY; i++) Setcols(colorwheel,255,	   255.0f*((float)i/(float)RY),	 0,	       k++);
    for (i = 0; i < YG; i++) Setcols(colorwheel,255.0f-255.0f*((float)i/(float)YG), 255,		 0,	       k++);
    for (i = 0; i < GC; i++) Setcols(colorwheel,0,		   255,		 255.0f*((float)i/(float)GC),     k++);
    for (i = 0; i < CB; i++) Setcols(colorwheel,0,		   255.0f-255.0f*((float)i/(float)CB), 255,	       k++);
    for (i = 0; i < BM; i++) Setcols(colorwheel,255.0f*((float)i/(float)BM),	   0,		 255,	       k++);
    for (i = 0; i < MR; i++) Setcols(colorwheel,255,	   0,		 255.0f-255.0f*((float)i/(float)MR), k++);


    //for (i=0;i<ncols;i++)
    //	cout<<"r "<<colorwheel[i*3]<<" g "<<colorwheel[i*3+1]<<" b "<<colorwheel[i*3+2]<<endl;
    return colorwheel;
}
//--------------------------------------------------------------------------
void TCVUtils::OpticalFlowToColor(TGpuMem::TGpuMemHalfFloat * MemU,TGpuMem::TGpuMemHalfFloat * MemV, TGpuMem::TGpuMemUChar *MemDst,float maxRad)
{

   if (MemDst->Channels()==3)
   {
	   //----------------------------------------------------------------------------------------------------
	   // Estimate the number of Blocks and number Threads
	   //----------------------------------------------------------------------------------------------------
       dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
       dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemU->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemU->Height(), numThreads.y));
	   //----------------------------------------------------------------------------------------------------

	   TGpuMem::TGpuMemHalfFloat *MemAux = new TGpuMem::TGpuMemHalfFloat(Gpu,MemU->Width(),MemU->Height(),1);
	   ((TGpu *)Gpu)->CV->Math->Eucliden_Norm(MemU,MemV,MemAux);

       float Max,Min,Avg;

       ((TGpu *)Gpu)->CV->Math->MaxMinAvg(MemAux,Max,Min,Avg);
       maxRad=Max;
       OpticalFlowToColor_Kernel<<<numBlocks, numThreads>>>(MemU->GetMemory(),MemV->GetMemory(), MemDst->GetMemory(),MemColorWheel->GetMemory(), maxRad,ncols,MemU->Width(),MemU->Height());
       cudaThreadSynchronize();

       //maxRad=19;
       cout<<"Max..."<<Max<<endl;
       cout<<"Min..."<<Min<<endl;
       cout<<"Avg..."<<Avg<<endl;

       //maxRad=maxRad+maxRad/10;

       delete MemAux;
   }
   else throw std::invalid_argument("MemDst must have 3 channels");

}
//--------------------------------------------------------------------------
void TCVUtils::StereoToColor(TGpuMem::TGpuMemHalfFloat * MemU, TGpuMem::TGpuMemUChar *MemDst)
{

   if (MemDst->Channels()==1)
   {
	   //----------------------------------------------------------------------------------------------------
	   // Estimate the number of Blocks and number Threads
	   //----------------------------------------------------------------------------------------------------
       dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
       dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(MemU->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(MemU->Height(), numThreads.y));
	   //----------------------------------------------------------------------------------------------------

	   TGpuMem::TGpuMemHalfFloat *MemAux1 = new TGpuMem::TGpuMemHalfFloat(Gpu,MemU->Width(),MemU->Height(),1);
	   TGpuMem::TGpuMemHalfFloat *MemAux2 = new TGpuMem::TGpuMemHalfFloat(Gpu,MemU->Width(),MemU->Height(),1);

	   ((TGpu *)Gpu)->CV->Math->Abs(MemU,MemAux1);
       float Max,Min,Avg;
  	   ((TGpu *)Gpu)->CV->Math->MaxMinAvg(MemAux1,Max,Min,Avg);
       ((TGpu *)Gpu)->CV->Math->Range(MemAux1,MemAux2,255.0,0.0);

       MemAux2->Casting(MemDst);
       //StereoToColor_Kernel<<<numBlocks, numThreads>>>(MemAux->GetMemory(), MemDst->GetMemory(),MemColorWheel->GetMemory(),ncols,MemU->Width(),MemU->Height());
       //cudaThreadSynchronize();

       cout<<"Max..."<<Max<<endl;
       cout<<"Min..."<<Min<<endl;
       cout<<"Avg..."<<Avg<<endl;

       delete MemAux1;
       delete MemAux2;
   }
   else throw std::invalid_argument("MemDst must have 3 channels");

}
//--------------------------------------------------------------------------
TCVUtils::~TCVUtils()
{
	delete MemColorWheel;
}
//--------------------------------------------------------------------------

