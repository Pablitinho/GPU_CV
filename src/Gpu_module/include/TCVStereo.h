/*
 * TCVStereo.h
 *
 *  Created on: 06/03/2015
 *      Author: pablo
 */

#ifndef TCVSTEREO_H_
#define TCVSTEREO_H_
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "TGpuMem.h"
#include "types_cc.h"

class SHARED_EXPORT TCVStereo {

public:
	TCVStereo(void * d_Gpu);
	void InitPyramid(int NumScalesMax,int MinSize,float Factor,uint Width, uint Height);
	void GeneratePyramid(TGpuMem::TGpuMemUChar  * MemSrc,TGpuMem::TGpuMemHalfFloat  ** MemPyramid);
	void GeneratePyramid(TGpuMem::TGpuMemHalfFloat  * MemSrc,TGpuMem::TGpuMemHalfFloat  ** MemPyramid);
	void GeneratePyramidOF(TGpuMem::TGpuMemHalfFloat  * MemSrc,TGpuMem::TGpuMemHalfFloat  ** MemPyramid);
	void AniTVL1_Stereo(TGpuMem::TGpuMemHalfFloat *MemSrc1,TGpuMem::TGpuMemHalfFloat *MemSrc2,TGpuMem::TGpuMemHalfFloat *U,int NumIter,int NumWarp,float alpha, float beta,bool PyramidalIter);
    void Compute_Census_Derivates(int NumScale, TGpuMem::TGpuMemHalfFloat  * _MemIm1,TGpuMem::TGpuMemHalfFloat  * _MemIm2, TGpuMem::TGpuMemHalfFloat *_MemIx, TGpuMem::TGpuMemHalfFloat *_MemIy, TGpuMem::TGpuMemHalfFloat *_MemIt);
	void InitPyramid();
	int NumScaleMaximum(uint Width, uint Height,int NumMax,int MinSize,float Factor);
	int NumScales(){return Scales;};
	void OpticalFlowToColor(TGpuMem::TGpuMemHalfFloat * MemU,TGpuMem::TGpuMemHalfFloat * MemV, TGpuMem::TGpuMemUChar *MemDst,float maxRad);
	void OpticalFlow_Threshold(TGpuMem::TGpuMemHalfFloat * MemUSrc,TGpuMem::TGpuMemHalfFloat * MemVSrc,TGpuMem::TGpuMemHalfFloat * MemUDst,TGpuMem::TGpuMemHalfFloat * MemVDst,float Value,bool Positive);
	~TCVStereo();

	TGpuMem::TGpuMemHalfFloat  ** MemPyrIm1;
	TGpuMem::TGpuMemHalfFloat  ** MemPyrIm2;
private:
	void Compute_OF_TV_L1_Huber(int NumScale,int NumIters, int NumWarps, float Alpha, float Beta, bool PiramidalIteration);
	void Compute_PEigen(int NumScale,bool FirsTime);
	void Update_OF_Up(int NumScale,int FirstTime,float Theta,float Sigma,int Warped);
	void Compute_Warping(int NumScale);

private:

	TGpuMem::TGpuMemHalfFloat  ** MemUPrev;
	TGpuMem::TGpuMemHalfFloat  ** MemVPrev;

	TGpuMem::TGpuMemHalfFloat  ** MemPyrIm2Warped;
	TGpuMem::TGpuMemHalfFloat  ** MemPyrG;

	TGpuMem::TGpuMemHalfFloat  ** MemP1;
	TGpuMem::TGpuMemHalfFloat  ** MemP2;

	TGpuMem::TGpuMemHalfFloat  ** MemAPP1;
	TGpuMem::TGpuMemHalfFloat  ** MemAPP2;

	TGpuMem::TGpuMemHalfFloat  ** MemDivx;

	TGpuMem::TGpuMemUInt  ** MemCensus1;
	TGpuMem::TGpuMemUInt  ** MemCensus2;

	TGpuMem::TGpuMemHalfFloat  ** MemIx;
	TGpuMem::TGpuMemHalfFloat  ** MemIy;
	TGpuMem::TGpuMemHalfFloat  ** MemIt;

	TGpuMem::TGpuMemHalfFloat  ** MemU;

	TGpuMem::TGpuMemHalfFloat  ** MemU0;

	TGpuMem::TGpuMemHalfFloat  ** MemUp;

	TGpuMem::TGpuMemHalfFloat  ** MemNx;
	TGpuMem::TGpuMemHalfFloat  ** MemNy;

	TGpuMem::TGpuMemHalfFloat  ** MemHFAux1;
	TGpuMem::TGpuMemHalfFloat  ** MemHFAux2;
	TGpuMem::TGpuMemHalfFloat  ** MemHFAux3;

private:
	void * Gpu;
	int Scales;
};
#include "TCV.h"
#include "TGpu.h"
#endif /* TCVSTEREO_H_ */
