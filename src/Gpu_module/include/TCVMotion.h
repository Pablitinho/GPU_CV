/*
 * TCVMotion.h
 *
 *  Created on: 28/02/2015
 *      Author: pablo
 */

#ifndef TCVMOTION_H_
#define TCVMOTION_H_
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "TGpuMem.h"
#include "types_cc.h"

class SHARED_EXPORT TCVMotion
{
public:
	TCVMotion(void * d_Gpu);
	void InitPyramid(int NumScalesMax,int MinSize,float Factor,uint Width, uint Height);
	void GeneratePyramid(TGpuMem::TGpuMemUChar  * MemSrc,TGpuMem::TGpuMemHalfFloat  ** MemPyramid);
	void GeneratePyramid(TGpuMem::TGpuMemHalfFloat  * MemSrc,TGpuMem::TGpuMemHalfFloat  ** MemPyramid);
	void GeneratePyramidOF(TGpuMem::TGpuMemHalfFloat  * MemSrc,TGpuMem::TGpuMemHalfFloat  ** MemPyramid);
	void AniTVL1(TGpuMem::TGpuMemHalfFloat *MemSrc1,TGpuMem::TGpuMemHalfFloat *MemSrc2,TGpuMem::TGpuMemHalfFloat *U,TGpuMem::TGpuMemHalfFloat *V,int NumIter,int NumWarp,float alpha, float beta,bool PyramidalIter);
	void AniTVL1(TGpuMem::TGpuMemHalfFloat *MemSrc1,TGpuMem::TGpuMemHalfFloat *MemSrc2,TGpuMem::TGpuMemHalfFloat *U,TGpuMem::TGpuMemHalfFloat *V,TGpuMem::TGpuMemHalfFloat *U_Prev,TGpuMem::TGpuMemHalfFloat *V_Prev,int NumIters,int NumWarps,float Alpha, float Beta,float PrevOFFactor,bool PyramidalIter);
	void Compute_Census_Derivates(int NumScale, TGpuMem::TGpuMemHalfFloat  * _MemIm1,TGpuMem::TGpuMemHalfFloat  * _MemIm2, TGpuMem::TGpuMemHalfFloat *_MemIx, TGpuMem::TGpuMemHalfFloat *_MemIy, TGpuMem::TGpuMemHalfFloat *_MemIt);
	void InitPyramid();
	int NumScales(){return Scales;};
    void OpticalFlow_Threshold(TGpuMem::TGpuMemHalfFloat * MemUSrc,TGpuMem::TGpuMemHalfFloat * MemVSrc,TGpuMem::TGpuMemHalfFloat * MemUDst,TGpuMem::TGpuMemHalfFloat * MemVDst,float Value,bool Positive);
	~TCVMotion();

	TGpuMem::TGpuMemHalfFloat  ** MemPyrIm1;
	TGpuMem::TGpuMemHalfFloat  ** MemPyrIm2;

private:

	void Compute_OF_TV_L1_Huber(int NumScale,int NumIters, int NumWarps, float Alpha, float Beta, bool PiramidalIteration);
	void Compute_OF_TV_L1_Huber_Prev(int NumScale, int NumIters, int NumWarps, float Alpha, float Beta, bool PiramidalIteration,float PrevOFFactor);
	void Compute_PEigen(int NumScale,bool FirsTime);
	void Update_OF_Up_Vp(int NumScale,int FirstTime,float Theta,float Sigma,int Warped);
	void Update_OF_Up_Vp_Prev(int NumScale,int FirstTime,float Theta,float Sigma,int Warped,float PrevOFFactor);
	void Iter_Compute_PEigen(int NumScale,bool FirstTime);
	void Compute_Warping(int NumScale);

private:

	TGpuMem::TGpuMemHalfFloat  ** MemUPrev;
	TGpuMem::TGpuMemHalfFloat  ** MemVPrev;

	TGpuMem::TGpuMemHalfFloat  ** MemPyrIm2Warped;
	TGpuMem::TGpuMemHalfFloat  ** MemPyrG;

	TGpuMem::TGpuMemHalfFloat  ** MemP1;
	TGpuMem::TGpuMemHalfFloat  ** MemP2;
	TGpuMem::TGpuMemHalfFloat  ** MemP3;
	TGpuMem::TGpuMemHalfFloat  ** MemP4;

	TGpuMem::TGpuMemHalfFloat  ** MemAPP1;
	TGpuMem::TGpuMemHalfFloat  ** MemAPP2;
	TGpuMem::TGpuMemHalfFloat  ** MemAPP3;
	TGpuMem::TGpuMemHalfFloat  ** MemAPP4;

	TGpuMem::TGpuMemHalfFloat  ** MemDivx;
	TGpuMem::TGpuMemHalfFloat  ** MemDivy;

	TGpuMem::TGpuMemUInt  ** MemCensus1;
	TGpuMem::TGpuMemUInt  ** MemCensus2;

	TGpuMem::TGpuMemHalfFloat  ** MemIx;
	TGpuMem::TGpuMemHalfFloat  ** MemIy;
	TGpuMem::TGpuMemHalfFloat  ** MemIt;

	TGpuMem::TGpuMemHalfFloat  ** MemU;
	TGpuMem::TGpuMemHalfFloat  ** MemV;

	TGpuMem::TGpuMemHalfFloat  ** MemU0;
	TGpuMem::TGpuMemHalfFloat  ** MemV0;

	TGpuMem::TGpuMemHalfFloat  ** MemUp;
	TGpuMem::TGpuMemHalfFloat  ** MemVp;

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


#endif
