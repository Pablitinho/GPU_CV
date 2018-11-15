
#include "TGpuMem.h"
#include "TGpu.h"
#include "TMathUtils.h"
#include "types_cc.h"
//==========================================================================
// Kernels
//==========================================================================
__global__ void CastingInt_Kernel(void * img_src, void * img_dst, uint src_type,uint dst_type, int Width, int Height)
{   //------------------------------------------------------------------
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int OffsetGray = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
	{
		switch(dst_type)
		{
		    case 0: {
		    	     int *optr= (int*) img_dst;
		    	     int *iptr= (int*) img_src;
		    	     optr[OffsetGray] = (int)iptr[OffsetGray];
		    	     break;
		            }
		    case 1: {//float
		    	     float *optr= (float*) img_dst;
		    	     int *iptr= (int*) img_src;
		    	     optr[OffsetGray] = (float)iptr[OffsetGray];
		    	     break;
		            }
		    case 2: {
		    	     unsigned char *optr= (unsigned char*) img_dst;
		    	     int *iptr= (int*) img_src;

		    	     optr[OffsetGray] = (unsigned char)iptr[OffsetGray];
		    	     break;
		            }
		    case 3: {
		    	     unsigned short *optr= (unsigned short*) img_dst;
		    	     int *iptr= (int*) img_src;
		    	     optr[OffsetGray] = __float2half_rn ((float)iptr[OffsetGray]);
		    	     break;
		            }
		    case 4: {

		    	     double *optr= (double*) img_dst;
		    	     int *iptr= (int*) img_src;
		    	     optr[OffsetGray] = (double)iptr[OffsetGray];
		    	     break;
		            }
		    case 5: {
		    	     unsigned int *optr= (unsigned int*) img_dst;
		    	     int *iptr= (int*) img_src;
		    	     optr[OffsetGray] = (unsigned int)iptr[OffsetGray];
		    	     break;
		            }
		}
	}
}
//-----------------------------------------------------------------------------------------
__global__ void CastingFloat_Kernel(void * img_src, void * img_dst, uint src_type,uint dst_type, int Width, int Height)
{   //------------------------------------------------------------------
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int OffsetGray = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
    {
    		switch(dst_type)
    		{
    		    case 0: {
    		    	     int *optr= (int*) img_dst;
    		    	     float *iptr= (float*) img_src;
    		    	     optr[OffsetGray] = (int)iptr[OffsetGray];
    		    	     break;
    		            }
    		    case 1: {//float
    		    	     float *optr= (float*) img_dst;
    		    	     float *iptr= (float*) img_src;
    		    	     optr[OffsetGray] = (float)iptr[OffsetGray];
    		    	     break;
    		            }
    		    case 2: {
    		    	     unsigned char *optr= (unsigned char*) img_dst;
    		    	     float *iptr= (float*) img_src;

    		    	     optr[OffsetGray] = (unsigned char)iptr[OffsetGray];
    		    	     break;
    		            }
    		    case 3: {
    		    	     unsigned short *optr= (unsigned short*) img_dst;
    		    	     float *iptr= (float*) img_src;
    		    	     optr[OffsetGray] = __float2half_rn (iptr[OffsetGray]);
    		    	     break;
    		            }
    		    case 4: {

    		    	     double *optr= (double*) img_dst;
    		    	     float *iptr= (float*) img_src;
    		    	     optr[OffsetGray] = (double)iptr[OffsetGray];
    		    	     break;
    		            }
    		    case 5: {
    		    	     unsigned int *optr= (unsigned int*) img_dst;
    		    	     float *iptr= (float*) img_src;
    		    	     optr[OffsetGray] = (unsigned int)iptr[OffsetGray];
    		    	     break;
    		            }
    		}
    	}
    }
//-----------------------------------------------------------------------------------------
__global__ void CastingUChar_Kernel(void * img_src, void * img_dst, uint src_type,uint dst_type, int Width, int Height)
{   //------------------------------------------------------------------
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int OffsetGray = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
    {
		switch(dst_type)
		{
		    case 0: {
		    	     int *optr= (int*) img_dst;
		    	     unsigned char *iptr= (unsigned char*) img_src;
		    	     optr[OffsetGray] = (int)iptr[OffsetGray];
		    	     break;
		            }
		    case 1: {//float
		    	     float *optr= (float*) img_dst;
		    	     unsigned char *iptr= (unsigned char*) img_src;
		    	     optr[OffsetGray] = (float)iptr[OffsetGray];
		    	     break;
		            }
		    case 2: {
		    	     unsigned char *optr= (unsigned char*) img_dst;
		    	     unsigned char *iptr= (unsigned char*) img_src;
		    	     optr[OffsetGray] = (unsigned char)iptr[OffsetGray];
		    	     break;
		            }
		    case 3: {
		    	     unsigned short *optr= (unsigned short*) img_dst;
		    	     unsigned char *iptr= (unsigned char*) img_src;
		    	     optr[OffsetGray] = __float2half_rn ((float)iptr[OffsetGray]);
		    	     break;
		            }
		    case 4: {

		    	     double *optr= (double*) img_dst;
		    	     unsigned char *iptr= (unsigned char*) img_src;
		    	     optr[OffsetGray] = (double)iptr[OffsetGray];
		    	     break;
		            }
		    case 5: {
		    	     unsigned int *optr= (unsigned int*) img_dst;
		    	     unsigned char *iptr= (unsigned char*) img_src;
		    	     optr[OffsetGray] = (unsigned int)iptr[OffsetGray];
		    	     break;
		            }
		}
	}
}
//-----------------------------------------------------------------------------------------
__global__ void CastingHalfFloat_Kernel(void * img_src, void * img_dst, uint src_type,uint dst_type, int Width, int Height)
{   //------------------------------------------------------------------
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int OffsetGray = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
    {
		switch(dst_type)
		{
		    case 0: {
		    	     int *optr= (int*) img_dst;
		    	     unsigned short *iptr= (unsigned short*) img_src;
		    	     optr[OffsetGray] = (int)__half2float(iptr[OffsetGray]);
		    	     break;
		            }
		    case 1: {//float
		    	     float *optr= (float*) img_dst;
		    	     unsigned short *iptr= (unsigned short*) img_src;
		    	     optr[OffsetGray] = (float)__half2float(iptr[OffsetGray]);
		    	     break;
		            }
		    case 2: {
		    	     unsigned char *optr= (unsigned char*) img_dst;
		    	     unsigned short *iptr= (unsigned short*) img_src;

		    	     optr[OffsetGray] = (unsigned char)__half2float(iptr[OffsetGray]);
		    	     break;
		            }
		    case 3: {
		    	     unsigned short *optr= (unsigned short*) img_dst;
		    	     unsigned short *iptr= (unsigned short*) img_src;
		    	     optr[OffsetGray] = iptr[OffsetGray];
		    	     break;
		            }
		    case 4: {

		    	     double *optr= (double*) img_dst;
		    	     unsigned short *iptr= (unsigned short*) img_src;
		    	     optr[OffsetGray] = (double)__half2float(iptr[OffsetGray]);
		    	     break;
		            }
		    case 5: {
		    	     unsigned int *optr= (unsigned int*) img_dst;
		    	     unsigned short *iptr= (unsigned short*) img_src;
		    	     optr[OffsetGray] = (unsigned int)__half2float(iptr[OffsetGray]);
		    	     break;
		            }
		}
	}
}
//-----------------------------------------------------------------------------------------
__global__ void CastingDouble_Kernel(void * img_src, void * img_dst, uint src_type,uint dst_type, int Width, int Height)
{   //------------------------------------------------------------------
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int OffsetGray = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
    {
		switch(dst_type)
		{
		    case 0: {
		    	     int *optr= (int*) img_dst;
		    	     double *iptr= (double*) img_src;
		    	     optr[OffsetGray] = (int)iptr[OffsetGray];
		    	     break;
		            }
		    case 1: {//float
		    	     float *optr= (float*) img_dst;
		    	     double *iptr= (double*) img_src;
		    	     optr[OffsetGray] = (float)iptr[OffsetGray];
		    	     break;
		            }
		    case 2: {
		    	     unsigned char *optr= (unsigned char*) img_dst;
		    	     double *iptr= (double*) img_src;
		    	     optr[OffsetGray] = (unsigned char)iptr[OffsetGray];
		    	     break;
		            }
		    case 3: {
		    	     unsigned short *optr= (unsigned short*) img_dst;
		    	     double *iptr= (double*) img_src;
		    	     optr[OffsetGray] = __float2half_rn ((float)iptr[OffsetGray]);
		    	     break;
		            }
		    case 4: {

		    	     double *optr= (double*) img_dst;
		    	     double *iptr= (double*) img_src;
		    	     optr[OffsetGray] = (double)iptr[OffsetGray];
		    	     break;
		            }
		    case 5: {
		    	     unsigned int *optr= (unsigned int*) img_dst;
		    	     double *iptr= (double*) img_src;
		    	     optr[OffsetGray] = (unsigned int)iptr[OffsetGray];
		    	     break;
		            }
		}
	}
}
//-----------------------------------------------------------------------------------------
__global__ void CastingUint_Kernel(void * img_src, void * img_dst, uint src_type,uint dst_type, int Width, int Height)
{   //------------------------------------------------------------------
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int OffsetGray = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
    {
		switch(dst_type)
		{
		    case 0: {
		    	     int *optr= (int*) img_dst;
		    	     unsigned int *iptr= (unsigned int*) img_src;
		    	     optr[OffsetGray] = (int)iptr[OffsetGray];
		    	     break;
		            }
		    case 1: {//float
		    	     float *optr= (float*) img_dst;
		    	     unsigned int *iptr= (unsigned int*) img_src;
		    	     optr[OffsetGray] = (float)iptr[OffsetGray];
		    	     break;
		            }
		    case 2: {
		    	     unsigned char *optr= (unsigned char*) img_dst;
		    	     unsigned int *iptr= (unsigned int*) img_src;
		    	     optr[OffsetGray] = (unsigned char)iptr[OffsetGray];
		    	     break;
		            }
		    case 3: {
		    	     unsigned short *optr= (unsigned short*) img_dst;
		    	     unsigned int *iptr= (unsigned int*) img_src;
		    	     optr[OffsetGray] = __float2half_rn ((float)iptr[OffsetGray]);
		    	     break;
		            }
		    case 4: {

		    	     double *optr= (double*) img_dst;
		    	     unsigned int *iptr= (unsigned int*) img_src;
		    	     optr[OffsetGray] = (double)iptr[OffsetGray];
		    	     break;
		            }
		    case 5: {
		    	     unsigned int *optr= (unsigned int*) img_dst;
		    	     unsigned int *iptr= (unsigned int*) img_src;
		    	     optr[OffsetGray] = (unsigned int)iptr[OffsetGray];
		    	     break;
		            }
		}
	}
}
//-----------------------------------------------------------------------------------------
__global__ void Init_int_Kernel(int * img_dst, int Value, int Width, int Height)
{
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
    {
		img_dst[GlobalOffset]=Value;
	}
}
//-----------------------------------------------------------------------------------------
__global__ void Init_float_Kernel(float * img_dst, float Value, int Width, int Height)
{
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int GlobalOffset = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
    {
		img_dst[GlobalOffset]=Value;
	}
}
//-----------------------------------------------------------------------------------------
__global__ void Init_half_float_Kernel(unsigned short * img_dst, float Value, int Width, int Height)
{
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int OffsetIm = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
    {
		img_dst[OffsetIm]=__float2half_rn(Value);
	}
}
//-----------------------------------------------------------------------------------------
__global__ void Init_double_Kernel(double * img_dst, double Value, int Width, int Height)
{
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int OffsetIm = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
    {
		img_dst[OffsetIm]=Value;
	}
}
//-----------------------------------------------------------------------------------------
__global__ void Init_uint_Kernel(unsigned int * img_dst, unsigned int Value, int Width, int Height)
{
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int OffsetIm = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
    {
		img_dst[OffsetIm]=Value;
	}
}
//-----------------------------------------------------------------------------------------
__global__ void Init_uchar_Kernel(unsigned char * img_dst, unsigned char Value, int Width, int Height)
{
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int OffsetIm = (globalY * Width + globalX);
    //------------------------------------------------------------------
	if (globalX>=0 && globalX<(Width) && globalY>=0 && globalY<(Height))
    {
		img_dst[OffsetIm]=Value;
	}
}
//-----------------------------------------------------------------------------------------
//==========================================================================

// End Kernels

//==========================================================================

//==========================================================================

// Int

//==========================================================================
TGpuMem::TGpuMemInt::TGpuMemInt(void *Gpu_,uint Width =0,uint Height=0,uint Channels=0):TGpuCoreMem(Gpu_, Width, Height, Channels, TGpuMem::t_uint)
{

}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemInt::CopyToDevice(int *h_Mem)
{
	TGpuMem::TGpuCoreMem::CopyToDevice((void *)h_Mem);
}
//-----------------------------------------------------------------------------------------
int * TGpuMem::TGpuMemInt::CopyFromDevice()
{
   return (int *) TGpuMem::TGpuCoreMem::CopyFromDevice();
}
//-----------------------------------------------------------------------------------------
int * TGpuMem::TGpuMemInt::GetMemory()
{
	return (int *) this->GetMem();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemInt::SetMem(int * Mem_)
{
	TGpuCoreMem::SetMem((void *) Mem_);
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemInt::Init(int value)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(d_Width, numThreads.x), ((TGpu *)Gpu)->iDivUp( d_Height, numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Init_int_Kernel<<<numBlocks, numThreads>>>((int *)d_Mem, value, d_Width, d_Height);
    cudaThreadSynchronize();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemInt::Copy(TGpuMemInt * MemDst)
{
	if (d_Size==MemDst->d_Size)
	    cudaMemcpy(MemDst->d_Mem, d_Mem, sizeof(int) * d_Size, cudaMemcpyDeviceToDevice);
	else
		throw std::invalid_argument("Different vectors' size");
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemInt::WriteFileFromDevice(const char * File)
{
	TMathUtils * myUtils = new TMathUtils();
	ofstream outfile;
    outfile.open(File);
    int * h_Mem = (  int*)CopyFromDevice();

    if (outfile.is_open())
    {
		for (uint r=0;r<d_Height;r++)
		{
			for (uint c=0;c<d_Width;c++)
			{
				outfile << h_Mem[r*d_Width+c];
				outfile << " ";
				//cout << (int)h_Mem[r*d_Width+c];
			}
			outfile << endl;
		}
		outfile.close();
    }
	delete h_Mem;
}
//-----------------------------------------------------------------------------------------
TGpuMem::TGpuMemInt::~TGpuMemInt()
{

}
//==========================================================================

// UInt

//==========================================================================
TGpuMem::TGpuMemUInt::TGpuMemUInt(void *Gpu_,uint Width =0,uint Height=0,uint Channels=0):TGpuCoreMem(Gpu_, Width, Height, Channels, TGpuMem::t_uint)
{

}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemUInt::CopyToDevice(unsigned int *h_Mem)
{
	TGpuMem::TGpuCoreMem::CopyToDevice((void *)h_Mem);
}
//-----------------------------------------------------------------------------------------
uint * TGpuMem::TGpuMemUInt::CopyFromDevice()
{
   return (uint *) TGpuMem::TGpuCoreMem::CopyFromDevice();
}
//-----------------------------------------------------------------------------------------
uint * TGpuMem::TGpuMemUInt::GetMemory()
{
	return (uint *) this->GetMem();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemUInt::SetMem(uint * Mem_)
{
	TGpuCoreMem::SetMem((void *) Mem_);
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemUInt::Init(unsigned int value)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(d_Width, numThreads.x), ((TGpu *)Gpu)->iDivUp( d_Height, numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Init_uint_Kernel<<<numBlocks, numThreads>>>((unsigned int *)d_Mem, value, d_Width, d_Height);
    cudaThreadSynchronize();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemUInt::Copy(TGpuMemUInt * MemDst)
{
	if (d_Size==MemDst->d_Size)
	    cudaMemcpy(MemDst->d_Mem, d_Mem, sizeof(unsigned int) * d_Size, cudaMemcpyDeviceToDevice);
	else
		throw std::invalid_argument("Different vectors' size");
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemUInt::WriteFileFromDevice(const char * File)
{
	TMathUtils * myUtils = new TMathUtils();
	ofstream outfile;
    outfile.open(File);
    unsigned int * h_Mem = (unsigned int*)CopyFromDevice();

    if (outfile.is_open())
    {
		for (uint r=0;r<d_Height;r++)
		{
			for (uint c=0;c<d_Width;c++)
			{

				outfile << h_Mem[r*d_Width+c];
				outfile << " ";
				//cout << (int)h_Mem[r*d_Width+c];
			}
			outfile << endl;
		}
		outfile.close();
    }
	delete h_Mem;
}
//-----------------------------------------------------------------------------------------
TGpuMem::TGpuMemUInt::~TGpuMemUInt()
{

}
//-----------------------------------------------------------------------------------------
//==========================================================================

// Float

//==========================================================================
TGpuMem::TGpuMemFloat::TGpuMemFloat(void *Gpu_,uint Width =0,uint Height=0,uint Channels=0):TGpuCoreMem(Gpu_, Width, Height, Channels, TGpuMem::t_float)
{

}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemFloat::CopyToDevice(float *h_Mem)
{
	TGpuMem::TGpuCoreMem::CopyToDevice((void *)h_Mem);
}
//-----------------------------------------------------------------------------------------
float * TGpuMem::TGpuMemFloat::CopyFromDevice()
{
   return (float *) TGpuMem::TGpuCoreMem::CopyFromDevice();
}
//-----------------------------------------------------------------------------------------
float * TGpuMem::TGpuMemFloat::GetMemory()
{
	return (float *) this->GetMem();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemFloat::SetMem(float * Mem_)
{
	TGpuCoreMem::SetMem((void *) Mem_);
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemFloat::Init(float value)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(d_Width, numThreads.x), ((TGpu *)Gpu)->iDivUp( d_Height, numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Init_float_Kernel<<<numBlocks, numThreads>>>((float *)d_Mem, value, d_Width, d_Height);
    cudaThreadSynchronize();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemFloat::Copy(TGpuMemFloat * MemDst)
{
	if (d_Size==MemDst->d_Size)
	    cudaMemcpy(MemDst->d_Mem, d_Mem, sizeof(float) * d_Size, cudaMemcpyDeviceToDevice);
	else
		throw std::invalid_argument( "Different vectors' size" );
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemFloat::WriteFileFromDevice(const char * File)
{
	TMathUtils * myUtils = new TMathUtils();
	ofstream outfile;
    outfile.open(File);
    float * h_Mem = (float*)CopyFromDevice();

    if (outfile.is_open())
    {
		for (uint r=0;r<d_Height;r++)
		{
			for (uint c=0;c<d_Width;c++)
			{
				outfile << h_Mem[r*d_Width+c];
				outfile << " ";
				//cout << (int)h_Mem[r*d_Width+c];
			}
			outfile << endl;
		}
		outfile.close();
    }
	delete h_Mem;
}
//-----------------------------------------------------------------------------------------
TGpuMem::TGpuMemFloat::~TGpuMemFloat()
{

}
//==========================================================================

// Half float

//==========================================================================
TGpuMem::TGpuMemHalfFloat::TGpuMemHalfFloat(void *Gpu_,uint Width =0,uint Height=0,uint Channels=0):TGpuCoreMem(Gpu_, Width, Height, Channels, TGpuMem::t_half_float)
{

}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemHalfFloat::CopyToDevice(unsigned short *h_Mem)
{
	TGpuMem::TGpuCoreMem::CopyToDevice((void *)h_Mem);
}
//-----------------------------------------------------------------------------------------
/*unsigned short * TGpuMem::TGpuMemHalfFloat::CopyFromDevice()
{
   return (unsigned short *) TGpuMem::TGpuCoreMem::CopyFromDevice();
}*/
//-----------------------------------------------------------------------------------------
float * TGpuMem::TGpuMemHalfFloat::CopyFromDevice()
{
   TGpuMem::TGpuMemFloat *MemFloat=new TGpuMem::TGpuMemFloat(Gpu,this->Width(),this->Height(),this->Channels());
   this->Casting(MemFloat);
   //cout<<"xx..."<<endl;
   float * Vector= MemFloat->CopyFromDevice();
   //cout<<"yy..."<<endl;
   delete MemFloat;

   return Vector;
}
//-----------------------------------------------------------------------------------------
unsigned short * TGpuMem::TGpuMemHalfFloat::GetMemory()
{
	return (unsigned short *) TGpuCoreMem::GetMem();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemHalfFloat::SetMem(unsigned short * Mem_)
{
	TGpuCoreMem::SetMem((void *) Mem_);
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemHalfFloat::Init(float value)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(d_Width, numThreads.x), ((TGpu *)Gpu)->iDivUp( d_Height, numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Init_half_float_Kernel<<<numBlocks, numThreads>>>((unsigned short *)d_Mem, value, d_Width, d_Height);
    cudaThreadSynchronize();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemHalfFloat::Copy(TGpuMemHalfFloat * MemDst)
{
	if (d_Size==MemDst->d_Size)
	    cudaMemcpy(MemDst->d_Mem, d_Mem, sizeof(unsigned short) * d_Size, cudaMemcpyDeviceToDevice);
	else
		throw std::invalid_argument( "Different vectors' size" );
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemHalfFloat::WriteFileFromDevice(const char * File)
{
	TMathUtils * myUtils = new TMathUtils();
	ofstream outfile;
    outfile.open(File);
    TGpuMem::TGpuMemFloat * MemFloat= new TGpuMem::TGpuMemFloat(Gpu,Width(), Height(),1);

    Casting(MemFloat);
    float * h_Mem = MemFloat->CopyFromDevice();

    if (outfile.is_open())
    {
		for (uint r=0;r<d_Height;r++)
		{
			for (uint c=0;c<d_Width;c++)
			{
				outfile << h_Mem[r*d_Width+c];
				outfile << " ";
				//cout << (int)h_Mem[r*d_Width+c];
			}
			outfile << endl;
		}
		outfile.close();
    }
	delete h_Mem;
	delete MemFloat;
}
//-----------------------------------------------------------------------------------------
TGpuMem::TGpuMemHalfFloat::~TGpuMemHalfFloat()
{

}
//==========================================================================

// Unsigned Char

//==========================================================================
TGpuMem::TGpuMemUChar::TGpuMemUChar(void *Gpu_,uint Width =0,uint Height=0,uint Channels=0):TGpuCoreMem(Gpu_, Width, Height, Channels, TGpuMem::t_uchar)
{

}
//-----------------------------------------------------------------------------------------
unsigned char * TGpuMem::TGpuMemUChar::CopyFromDevice()
{
   return (unsigned char *) TGpuMem::TGpuCoreMem::CopyFromDevice();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemUChar::WriteFileFromDevice(const char * File)
{
	TMathUtils * myUtils = new TMathUtils();
	ofstream outfile;
    outfile.open(File);
    unsigned char * h_Mem = (unsigned char*)CopyFromDevice();

    if (outfile.is_open())
    {
		for (uint r=0;r<d_Height;r++)
		{
			for (uint c=0;c<d_Width;c++)
			{
			      outfile << (int)h_Mem[r*d_Width+c];

				outfile << " ";
				//cout << (int)h_Mem[r*d_Width+c];
			}
			outfile << endl;
		}
		outfile.close();
    }
	delete h_Mem;
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemUChar::CopyToDevice(unsigned char *h_Mem)
{
	TGpuMem::TGpuCoreMem::CopyToDevice((void *)h_Mem);
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemUChar::SetMem(unsigned char * Mem_)
{
	TGpuMem::TGpuCoreMem::SetMem((void *) Mem_);
}
//-----------------------------------------------------------------------------------------
unsigned char * TGpuMem::TGpuMemUChar::GetMemory()
{
	return (unsigned char *) TGpuMem::TGpuCoreMem::GetMem();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemUChar::Init(unsigned char value)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(d_Width, numThreads.x), ((TGpu *)Gpu)->iDivUp( d_Height, numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Init_uchar_Kernel<<<numBlocks, numThreads>>>((unsigned char *)d_Mem, value, d_Width, d_Height);
    cudaThreadSynchronize();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemUChar::Copy(TGpuMemUChar * MemDst)
{
	if (d_Size==MemDst->d_Size)
	    cudaMemcpy(MemDst->d_Mem, d_Mem, sizeof(unsigned char) * d_Size, cudaMemcpyDeviceToDevice);
	else
		throw std::invalid_argument( "Different vectors' size" );
}
//-----------------------------------------------------------------------------------------
TGpuMem::TGpuMemUChar::~TGpuMemUChar()
{

}
//==========================================================================
//==========================================================================

// Double

//==========================================================================
TGpuMem::TGpuMemDouble::TGpuMemDouble(void *Gpu_,uint Width =0,uint Height=0,uint Channels=0):TGpuCoreMem(Gpu_, Width, Height, Channels, TGpuMem::t_uchar)
{

}
//-----------------------------------------------------------------------------------------
double * TGpuMem::TGpuMemDouble::CopyFromDevice()
{
   return (double *) TGpuMem::TGpuCoreMem::CopyFromDevice();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemDouble::CopyToDevice(double *h_Mem)
{
	TGpuMem::TGpuCoreMem::CopyToDevice((void *)h_Mem);
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemDouble::SetMem(double * Mem_)
{
	TGpuMem::TGpuCoreMem::SetMem((void *) Mem_);
}
//-----------------------------------------------------------------------------------------
double * TGpuMem::TGpuMemDouble::GetMemory()
{
	return (double *) TGpuMem::TGpuCoreMem::GetMem();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemDouble::Init(double value)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(d_Width, numThreads.x), ((TGpu *)Gpu)->iDivUp( d_Height, numThreads.y));
	//----------------------------------------------------------------------------------------------------
    Init_double_Kernel<<<numBlocks, numThreads>>>((double *)d_Mem, value, d_Width, d_Height);
    cudaThreadSynchronize();
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemDouble::Copy(TGpuMemDouble * MemDst)
{
	if (d_Size==MemDst->d_Size)
	    cudaMemcpy(MemDst->d_Mem, d_Mem, sizeof(double) * d_Size, cudaMemcpyDeviceToDevice);
	else
		throw std::invalid_argument( "Different vectors' size" );
}
//-----------------------------------------------------------------------------------------
void TGpuMem::TGpuMemDouble::WriteFileFromDevice(const char * File)
{
	TMathUtils * myUtils = new TMathUtils();
	ofstream outfile;
    outfile.open(File);
    double * h_Mem = (double*)CopyFromDevice();

    if (outfile.is_open())
    {
		for (uint r=0;r<d_Height;r++)
		{
			for (uint c=0;c<d_Width;c++)
			{
			      outfile << h_Mem[r*d_Width+c];

				outfile << " ";
				//cout << (int)h_Mem[r*d_Width+c];
			}
			outfile << endl;
		}
		outfile.close();
    }
	delete h_Mem;
}
//-----------------------------------------------------------------------------------------
TGpuMem::TGpuMemDouble::~TGpuMemDouble()
{

}
//==========================================================================
// Core Mem

//==========================================================================
//--------------------------------------------------------------------------
TGpuMem::TGpuCoreMem::TGpuCoreMem(void *Gpu_,uint Width,uint Height,uint Channels,int MemType_)
{
	d_Size = Width*Height*Channels;
	d_Width = Width;
	d_Height = Height;
	d_Channels = Channels;
    Gpu=Gpu_;

	switch(MemType_)
    {
		    case t_int: //int
		    		{
		    	      cudaMalloc((void**) &d_Mem, sizeof(int) * d_Size);
		    	      break;
		            }
		    case t_float: //float
		    		{
		    	      cudaMalloc((void**) &d_Mem, sizeof(float) * d_Size);
		    	      break;
		            }
		    case t_uchar: //unsigned char
		    		{
			    	  cudaMalloc((void**) &d_Mem, sizeof(unsigned char) * d_Size);
			    	  break;
		            }
		    case t_half_float: //half_float
		    		{
			    	  cudaMalloc((void**) &d_Mem, sizeof(unsigned short) * d_Size);
			    	  break;
		            }
		    case t_double: //double
		            {
			    	  cudaMalloc((void**) &d_Mem, sizeof(double) * d_Size);
			    	  break;
		            }
		    case t_uint: //unsigned int
		    		{
			    	  cudaMalloc((void**) &d_Mem, sizeof(unsigned int) * d_Size);
			    	  break;
		            }
		}

	   d_MemType = MemType_;
}
//--------------------------------------------------------------------------
void TGpuMem::TGpuCoreMem::CopyToDevice(void *h_Mem)
{
	switch(d_MemType)
	{

		    case t_int: //int
		    		{
		    	      cudaMemcpy(d_Mem, h_Mem, sizeof(int) * d_Size, cudaMemcpyHostToDevice);
		    	      break;
		            }
		    case t_float: //float
		    		{
		    	      cudaMemcpy(d_Mem, h_Mem, sizeof(float) * d_Size, cudaMemcpyHostToDevice);
		    	      break;
		            }
		    case t_uchar: //unsigned char
		    		{

			    	  cudaMemcpy(d_Mem, h_Mem, sizeof(unsigned char) * d_Size, cudaMemcpyHostToDevice);
			    	  break;
		            }
		    case t_half_float: //half_float
		    		{
			    	  cudaMemcpy(d_Mem, h_Mem, sizeof(unsigned short) * d_Size, cudaMemcpyHostToDevice);
			    	  break;
		            }
		    case t_double: //double
		            {
			    	  cudaMemcpy(d_Mem, h_Mem, sizeof(double) * d_Size, cudaMemcpyHostToDevice);
			    	  break;
		            }
		    case t_uint: //unsigned int
		    		{
			    	  cudaMemcpy(d_Mem, h_Mem, sizeof(unsigned int) * d_Size, cudaMemcpyHostToDevice);
			    	  break;
		            }
		}
}
//--------------------------------------------------------------------------
void * TGpuMem::TGpuCoreMem::CopyFromDevice( )
{

	switch(d_MemType)
	{
		    case t_int: //int
		    		{
		    		  void * h_Mem= new int[d_Size];
		    		  cudaMemcpy(h_Mem, d_Mem, sizeof(int) * d_Size, cudaMemcpyDeviceToHost);
		    		  return h_Mem;

		            }
		    case t_float: //float
		    		{
			          void * h_Mem= new float[d_Size];
			          cudaMemcpy(h_Mem, d_Mem, sizeof(float) * d_Size, cudaMemcpyDeviceToHost);
			          return h_Mem;

		            }
		    case t_uchar: //unsigned char
		    		{
		    			cout << "-"<<endl;
				      void * h_Mem= new unsigned char[d_Size];
				      cudaMemcpy(h_Mem, d_Mem, sizeof(unsigned char) * d_Size, cudaMemcpyDeviceToHost);
				      return h_Mem;

		            }
		    case t_half_float: //half_float
		    		{
					  void * h_Mem= new unsigned short[d_Size];
					  cudaMemcpy(h_Mem, d_Mem, sizeof(unsigned short) * d_Size, cudaMemcpyDeviceToHost);
					  return h_Mem;

		            }
		    case t_double: //double
		            {
			          void * h_Mem= new double[d_Size];
					  cudaMemcpy(h_Mem, d_Mem, sizeof(double) * d_Size, cudaMemcpyDeviceToHost);
					  return h_Mem;

		            }
		    case t_uint: //unsigned int
		    		{
				      void * h_Mem= new unsigned int[d_Size];
				      cudaMemcpy(h_Mem, d_Mem, sizeof(unsigned int) * d_Size, cudaMemcpyDeviceToHost);
				      return h_Mem;

		            }
	}
	return NULL;

}
//--------------------------------------------------------------------------
/*void TGpuMem::TGpuCoreMem::WriteFileToDevice(const char * File)
{

	ifstream infile;
	infile.open(File);

	void * h_Mem = new T[d_Size];

    if (infile.is_open())
    {
		for (int r=0;r<d_Height;r++)
			for (int c=0;c<d_Width;c++)
				infile >> h_Mem[r*d_Width+c];

		infile.close();
    }
    CopyToDevice(h_Mem);
	delete h_Mem;

}*/
//--------------------------------------------------------------------------
void TGpuMem::TGpuCoreMem::Casting(TGpuCoreMem * dst_Mem)
{
	//----------------------------------------------------------------------------------------------------
	// Estimate the number of Blocks and number Threads
	//----------------------------------------------------------------------------------------------------
    dim3 numThreads = dim3(((TGpu *)Gpu)->GetBlockX(), ((TGpu *)Gpu)->GetBlockY(), 1);
    dim3 numBlocks =  dim3(((TGpu *)Gpu)->iDivUp(d_Width, numThreads.x), ((TGpu *)Gpu)->iDivUp(d_Height, numThreads.y));
	//----------------------------------------------------------------------------------------------------
	enum { t_int = 0, t_float = 1, t_uchar = 2, t_half_float = 3, t_double = 4, t_uint = 5};
    switch(d_MemType)
    {
    	case 0: {//int
    		      CastingInt_Kernel<<<numBlocks, numThreads>>>(d_Mem, dst_Mem->d_Mem, (int)d_MemType, (int)dst_Mem->d_MemType,d_Width, d_Height);
    		      break;
    		    }
    	case 1: {//float
    		      CastingFloat_Kernel<<<numBlocks, numThreads>>>(d_Mem, dst_Mem->d_Mem, (int)d_MemType, (int)dst_Mem->d_MemType,d_Width, d_Height);
    		      break;
    	        }
        case 2: {//uchar
        	      CastingUChar_Kernel<<<numBlocks, numThreads>>>(d_Mem, dst_Mem->d_Mem, (int)d_MemType, (int)dst_Mem->d_MemType,d_Width, d_Height);
    		      break;
    		    }
        case 3: {//half_float
        	      CastingHalfFloat_Kernel<<<numBlocks, numThreads>>>(d_Mem, dst_Mem->d_Mem, (int)d_MemType, (int)dst_Mem->d_MemType,d_Width, d_Height);
    		      break;
    		    }
        case 4: {//double
        	      CastingDouble_Kernel<<<numBlocks, numThreads>>>(d_Mem, dst_Mem->d_Mem, (int)d_MemType, (int)dst_Mem->d_MemType,d_Width, d_Height);
    		      break;
    		    }
        case 5: {//uint
        	      CastingUint_Kernel<<<numBlocks, numThreads>>>(d_Mem, dst_Mem->d_Mem, (int)d_MemType, (int)dst_Mem->d_MemType,d_Width, d_Height);
    		      break;
    		    }
    }

	cudaThreadSynchronize();
	//----------------------------------------------------------------------------------------------------
}
//--------------------------------------------------------------------------
TGpuMem::TGpuCoreMem::~TGpuCoreMem()
{
	(cudaFree((void*) d_Mem));
}
//--------------------------------------------------------------------------
