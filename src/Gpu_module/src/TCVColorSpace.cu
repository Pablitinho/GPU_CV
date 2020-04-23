#include "TCVColorSpace.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"
//==========================================================================
// Kernels
//==========================================================================
__global__ void RgbToGray_Kernel(unsigned char * RGB_Image, unsigned char * Gray_Image, int Width, int Height)
{   //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;
    int OffsetGray = (globalY * Width + globalX);
    int OffsetColor = (globalY * Width + globalX)*3;
    //------------------------------------------------------------------
	
    if(globalX<Width && globalY<Height)
    {
       Gray_Image[OffsetGray] = (unsigned char)(0.114f*RGB_Image[OffsetColor]+0.587f*RGB_Image[OffsetColor+1]+0.299f*RGB_Image[OffsetColor+2]);
    }
}
//==========================================================================
__global__ void RgbToGray_hf_Kernel(unsigned char * RGB_Image, unsigned short * Gray_Image, int Width, int Height)
{
    //------------------------------------------------------------------
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;
    int OffsetGray = (globalY * Width + globalX);
    int OffsetColor = (globalY * Width + globalX)*3;
    //------------------------------------------------------------------
    if(globalX>=0 && globalX<Width && globalY>=0 && globalY<Height)
    {
       Gray_Image[OffsetGray] = __float2half_rn((float)(0.114f*RGB_Image[OffsetColor]+0.587f*RGB_Image[OffsetColor+1]+0.299f*RGB_Image[OffsetColor+2]));
    }
}
//==========================================================================
// End Kernels
//==========================================================================
//--------------------------------------------------------------------------
//==========================================================================
// Class Methods
//==========================================================================
TCVColorSpace::TCVColorSpace(void * d_Gpu)
{
    Gpu = d_Gpu;
}
//--------------------------------------------------------------------------
void TCVColorSpace:: RgbToGray(TGpuMem::TGpuMemUChar * RGB_Image, TGpuMem::TGpuMemUChar * Gray_Image)
{
    //----------------------------------------------------------------------------------------------------
    // Estimate the number of Blocks and number Threads
    //----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(RGB_Image->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(RGB_Image->Height(), numThreads.y));
    //----------------------------------------------------------------------------------------------------

	printf("width: %i height %i", RGB_Image->Width(), RGB_Image->Height());
	unsigned char * mem = Gray_Image->GetMemory();
    RgbToGray_Kernel<<<numBlocks, numThreads>>>(RGB_Image->GetMemory(), Gray_Image->GetMemory(), RGB_Image->Width(), RGB_Image->Height());
    cudaThreadSynchronize();
    //----------------------------------------------------------------------------------------------------
}
//--------------------------------------------------------------------------
void TCVColorSpace:: RgbToGray(TGpuMem::TGpuMemUChar * RGB_Image,TGpuMem::TGpuMemHalfFloat * Gray_Image)
{
	((TGpu *)Gpu)->SetDevice(((TGpu *)Gpu)->GetDevice());
    //----------------------------------------------------------------------------------------------------
    // Estimate the number of Blocks and number Threads
    //----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(RGB_Image->Width(), numThreads.x), ((TGpu *)Gpu)->iDivUp(RGB_Image->Height(), numThreads.y));
    //----------------------------------------------------------------------------------------------------
    RgbToGray_hf_Kernel<<<numBlocks, numThreads>>>(RGB_Image->GetMemory(), Gray_Image->GetMemory(), RGB_Image->Width(), RGB_Image->Height());
    cudaThreadSynchronize();
    //----------------------------------------------------------------------------------------------------
    //((TGpu *)Gpu)->CV->Geometry->Test();
}
//--------------------------------------------------------------------------
TCVColorSpace::~TCVColorSpace()
{
    //delete CV;
}
//--------------------------------------------------------------------------
