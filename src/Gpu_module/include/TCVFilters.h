/*
 * TCVFilters.h
 *
 *  Created on: 23/02/2015
 *      Author: pablo
 */

#ifndef TCVFILTERS_H_
#define TCVFILTERS_H_

#include "TGpuMem.h"

class SHARED_EXPORT TCVFilters
{

public:
	TCVFilters(void * d_Gpu);
	void Median3x3(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst);
	void SeparableConvolution_H(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst,TGpuMem::TGpuMemFloat * MemFilter);
	void SeparableConvolution_V(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst,TGpuMem::TGpuMemFloat * MemFilter);
	void SeparableConvolution_H(TGpuMem::TGpuMemUChar * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst,TGpuMem::TGpuMemFloat * MemFilter);
	void SeparableConvolution_V(TGpuMem::TGpuMemUChar * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst,TGpuMem::TGpuMemFloat * MemFilter);
	void SeparableConvolution(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst,TGpuMem::TGpuMemFloat * MemFilter);
    void FirstOrderStructureTensor(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemIx2,TGpuMem::TGpuMemHalfFloat * MemIy2, TGpuMem::TGpuMemHalfFloat * MemIxIy,TGpuMem::TGpuMemFloat * MemFilter);
	void CensusDerivates(TGpuMem::TGpuMemUInt * MemCensus1,TGpuMem::TGpuMemUInt * MemCensus2, TGpuMem::TGpuMemHalfFloat * MemIx,TGpuMem::TGpuMemHalfFloat * MemIy, TGpuMem::TGpuMemHalfFloat * MemIt);
	~TCVFilters();

private:
	void * Gpu;

public:
    TGpuMem::TGpuMemFloat * d_FilterGauss_0_5;
    TGpuMem::TGpuMemFloat * d_FilterGauss_1;
    TGpuMem::TGpuMemFloat * d_FilterGauss_1_5;
    TGpuMem::TGpuMemFloat * d_FilterGauss_2;
    TGpuMem::TGpuMemFloat * d_FilterGauss_2_5;
    TGpuMem::TGpuMemFloat * d_FilterGauss_3;
    TGpuMem::TGpuMemFloat * d_FilterGauss_3_5;
    TGpuMem::TGpuMemFloat * d_FilterGauss_4;
    TGpuMem::TGpuMemFloat * d_FilterDerivate_5;
    TGpuMem::TGpuMemFloat * d_FilterGauss_x;

};
#include "TCV.h"
#include "TGpu.h"
#endif /* TCVFILTERS_H_ */
