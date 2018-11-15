/*
 * TCVUtils.h
 *
 *  Created on: 04/03/2015
 *      Author: pablo
 */

#ifndef TCVUTILS_H_
#define TCVUTILS_H_
#include "TGpuMem.h"
#include "types_cc.h"

class SHARED_EXPORT TCVUtils
{

public:
	TCVUtils(void * d_Gpu);
	void Padding(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst,uint Xoffset,uint Yoffset,float Value);
	void ReplicateEdges(TGpuMem::TGpuMemHalfFloat * MemU, TGpuMem::TGpuMemHalfFloat * MemV);
	void OpticalFlowToColor(TGpuMem::TGpuMemHalfFloat * MemU,TGpuMem::TGpuMemHalfFloat * MemV, TGpuMem::TGpuMemUChar *MemDst,float maxRad);
	void StereoToColor(TGpuMem::TGpuMemHalfFloat * MemU, TGpuMem::TGpuMemUChar *MemDst);
	~TCVUtils();

private:
	void * Gpu;
	int ncols;
	TGpuMem::TGpuMemInt  * MemColorWheel;
	void Setcols(int *colorwheel,int r, int g, int b, int k);
    int * Makecolorwheel();


};
#include "TCV.h"
#include "TGpu.h"

#endif /* TCVUTILS_H_ */
