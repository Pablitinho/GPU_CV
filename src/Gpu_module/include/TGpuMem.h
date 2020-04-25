#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <typeinfo>
#include <stdexcept>
#include "types_cc.h"
#include "cuda_fp16.h"
using namespace std;

#ifndef TGPUMEM_H_
#define TGPUMEM_H_
#include "shared_EXPORTS.h"


class SHARED_EXPORT TGpuMem
{

public:
	enum { t_int = 0, t_float = 1, t_uchar = 2, t_half_float = 3, t_double = 4, t_uint = 5};

	class SHARED_EXPORT TGpuCoreMem {

	//-----------------------------------
	public:

		TGpuCoreMem(void *Gpu_,uint Width ,uint Height,uint Channels,int MemType_,bool use_zero_copy_);

		void CopyToDevice(void *h_Mem);
		void * CopyFromDevice();
		//void WriteFileToDevice(const char * File);
		uint Width(){ return d_Width; };
		uint Height(){ return d_Height; };
		uint Channels(){return d_Channels;};
		uint Size(){ return d_Size; };
		void SetMem(void * Mem_){ d_Mem = Mem_; };
		int GetVarType(){return d_MemType;};
		void Casting(TGpuCoreMem * dst_Mem);
	    ~TGpuCoreMem();
	//-----------------------------------
	protected:
		void * GetMem()
		{
			return d_Mem;
		};
	    void *  d_Mem;
	    uint d_Size;
	    uint d_Width;
	    uint d_Height;
	    uint d_Channels;
	    int d_MemType;
	    void * Gpu;
		bool use_zero_copy;

	//-----------------------------------
	};
	class SHARED_EXPORT TGpuMemUChar : public TGpuCoreMem {
	//-----------------------------------
	public:

		TGpuMemUChar(void * Gpu_,uint Width,uint Height,uint Channels,bool use_zero_copy_);
		void CopyToDevice(unsigned char *h_Mem);
		void SetMem(unsigned char * Mem_);
		unsigned char * CopyFromDevice();
		unsigned char * GetMemory();
		void Init(unsigned char value);
		void Copy(TGpuMemUChar * MemDst);
		void WriteFileFromDevice(const char * File);
		~TGpuMemUChar();
	//-----------------------------------
	};
	//---------------------------------------------------------------------------------------------------------------------
	class SHARED_EXPORT TGpuMemInt : public TGpuCoreMem {
	//-----------------------------------
	public:

		TGpuMemInt(void *Gpu_,uint Width,uint Height,uint Channel, bool use_zero_copy_);
		void CopyToDevice(int *h_Mem);
		int * CopyFromDevice();
		void SetMem(int * Mem_);
		int * GetMemory();
		void Init(int value);
		void Copy(TGpuMemInt * MemDst);
		void WriteFileFromDevice(const char * File);
		~TGpuMemInt();
	//-----------------------------------
	};
	//---------------------------------------------------------------------------------------------------------------------
	class SHARED_EXPORT TGpuMemUInt : public TGpuCoreMem {
	//-----------------------------------
	public:

		TGpuMemUInt(void *Gpu_,uint Width,uint Height,uint Channels, bool use_zero_copy_);
		void CopyToDevice(unsigned int *h_Mem);
		unsigned int * CopyFromDevice();
		void SetMem(uint * Mem_);
		unsigned int* GetMemory();
		void Init(unsigned int value);
		void Copy(TGpuMemUInt * MemDst);
		void WriteFileFromDevice(const char * File);
		~TGpuMemUInt();
	//-----------------------------------
	};
	//---------------------------------------------------------------------------------------------------------------------
	class SHARED_EXPORT TGpuMemFloat : public TGpuCoreMem {
	//-----------------------------------
	public:

		TGpuMemFloat(void *Gpu_,uint Width,uint Height,uint Channels, bool use_zero_copy_);
		void CopyToDevice(float *h_Mem);
		float * CopyFromDevice();
		void SetMem(float * Mem_);
		float* GetMemory();
		void Init(float value);
		void Copy(TGpuMemFloat * MemDst);
		void WriteFileFromDevice(const char * File);
		~TGpuMemFloat();
	//-----------------------------------
	};
	//---------------------------------------------------------------------------------------------------------------------
	class SHARED_EXPORT TGpuMemHalfFloat : public TGpuCoreMem {
	//-----------------------------------
	public:

		TGpuMemHalfFloat(void *Gpu_,uint Width,uint Height,uint Channels, bool use_zero_copy_);
		void CopyToDevice(half *h_Mem);
		float *CopyFromDevice();
		void SetMem(half * Mem_);
		half* GetMemory();
		void Init(float value);
		void Copy(TGpuMemHalfFloat * MemDst);
		void WriteFileFromDevice(const char * File);
		~TGpuMemHalfFloat();
	//-----------------------------------
	};
	//---------------------------------------------------------------------------------------------------------------------
	class SHARED_EXPORT TGpuMemDouble : public TGpuCoreMem {
	//-----------------------------------
	public:

		TGpuMemDouble(void *Gpu_,uint Width,uint Height,uint Channels, bool use_zero_copy_);
		void CopyToDevice(double *h_Mem);
		double * CopyFromDevice();
		void SetMem(double * Mem_);
		double* GetMemory();
		void Init(double value);
		void Copy(TGpuMemDouble * MemDst);
		void WriteFileFromDevice(const char * File);
		~TGpuMemDouble();
	//-----------------------------------
	};
	//---------------------------------------------------------------------------------------------------------------------
};

#endif /* TGPUMEM_H_ */
