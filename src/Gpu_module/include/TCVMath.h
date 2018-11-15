/*
 * TCVMath.h
 *
 *  Created on: 19/02/2015
 *      Author: pablo
 */

#ifndef TCVMATH_H_
#define TCVMATH_H_

//class TGpuMem;
//class TGpu;
//class TCV;
#include "TGpuMem.h"
#include "types_cc.h"

class SHARED_EXPORT TCVMath {
public:
	TCVMath(void * d_Gpu);
	void Mult(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst);
	void Mult(TGpuMem::TGpuMemHalfFloat * MemSrc,float Value,TGpuMem::TGpuMemHalfFloat * MemDst);
	void Div(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst);
	void Div(TGpuMem::TGpuMemHalfFloat * MemSrc,float Value,TGpuMem::TGpuMemHalfFloat * MemDst);
	void Subtract(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst);
	void Eucliden_Norm(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst);
	void Subtract(TGpuMem::TGpuMemHalfFloat * MemSrc,float Value,TGpuMem::TGpuMemHalfFloat * MemDst);
	void Subtract(TGpuMem::TGpuMemUChar * MemSrc1,TGpuMem::TGpuMemUChar * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst);
	void SubtractAbs(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst);
	void HammingDistance(TGpuMem::TGpuMemUInt * MemSrc1,TGpuMem::TGpuMemUInt * MemSrc2, TGpuMem::TGpuMemHalfFloat * MemDst,float Factor);
	void Transpose(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst);
	void Divergence(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2,TGpuMem::TGpuMemHalfFloat * MemDst);
	void MaxMinAvg(TGpuMem::TGpuMemHalfFloat * MemoryInput,float &Max,float &Min,float &Avg);
	unsigned int MaxFactor(unsigned int number,unsigned int Maximum);
	int MaximumScales(uint Width, uint Height, int NumMax,int MinSize, float Factor);
	void Range(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat *MemDst,float MaxOut, float MinOut);
	void Abs(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat *MemDst);
	~TCVMath();

private:
	void * Gpu;

};
#include "TCV.h"
#include "TGpu.h"

#endif /* TCVMATH_H_ */
