/*
 * TCVFeatures.h
 *
 *  Created on: 26/02/2015
 *      Author: pablo
 */

#ifndef TCVFEATURES_H_
#define TCVFEATURES_H_
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "TGpuMem.h"


class SHARED_EXPORT TCVFeatures {
public:
	TCVFeatures(void * d_Gpu);
	void DiffusionWeight(TGpuMem::TGpuMemUChar  * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst, float alpha,float beta);
	void DiffusionWeight(TGpuMem::TGpuMemHalfFloat  * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst, float alpha,float beta);
	void Census(TGpuMem::TGpuMemUChar * MemSrc, TGpuMem::TGpuMemUInt * MemDst,int eps);
	void Census(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemUInt * MemDst,int eps);
	void Derivates(TGpuMem::TGpuMemUChar * MemSrc, TGpuMem::TGpuMemHalfFloat * Ix,TGpuMem::TGpuMemHalfFloat * Iy);
	void Derivates(TGpuMem::TGpuMemHalfFloat  * MemSrc, TGpuMem::TGpuMemHalfFloat * MemIx,TGpuMem::TGpuMemHalfFloat * MemIy);
	void Derivates(TGpuMem::TGpuMemUChar * MemSrc1,TGpuMem::TGpuMemUChar * MemSrc2, TGpuMem::TGpuMemHalfFloat * MemIx,TGpuMem::TGpuMemHalfFloat * MemIy,TGpuMem::TGpuMemHalfFloat * MemIt);
	void Derivates(TGpuMem::TGpuMemHalfFloat * MemSrc1,TGpuMem::TGpuMemHalfFloat * MemSrc2, TGpuMem::TGpuMemHalfFloat * MemIx,TGpuMem::TGpuMemHalfFloat * MemIy,TGpuMem::TGpuMemHalfFloat * MemIt);
	void EigenVectors(TGpuMem::TGpuMemUChar * MemSrc, TGpuMem::TGpuMemHalfFloat * MemNx,TGpuMem::TGpuMemHalfFloat * MemNy,TGpuMem::TGpuMemFloat * MemFilter);
	void EigenVectors(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemNx,TGpuMem::TGpuMemHalfFloat * MemNy,TGpuMem::TGpuMemFloat * MemFilter);
	~TCVFeatures();

private:
	void * Gpu;

};

#include "TCV.h"
#include "TGpu.h"

#endif /* TCVFEATURES_H_ */
