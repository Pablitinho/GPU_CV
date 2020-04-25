/*
 * TCVGeometry.h
 *
 *  Created on: 18/02/2015
 *      Author: pablo
 */

#ifndef TCVGEOMETRY_H_
#define TCVGEOMETRY_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

#include "TGpuMem.h"

class SHARED_EXPORT TCVGeometry {

public:
	TCVGeometry(void * d_Gpu);
	void Test(){cout<< "TEST"<<endl;};
	void Resize(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst);
	void ResizeBilinear(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst);
	void Resize(TGpuMem::TGpuMemUChar * MemSrc, TGpuMem::TGpuMemUChar * MemDst);
	void ResizeBilinear(TGpuMem::TGpuMemUChar * MemSrc, TGpuMem::TGpuMemUChar * MemDst);
	void Warping(TGpuMem::TGpuMemHalfFloat * MemSrc, TGpuMem::TGpuMemHalfFloat * MemDst, TGpuMem::TGpuMemHalfFloat * MemU, TGpuMem::TGpuMemHalfFloat * MemV,bool Inverted);
	void Warping(TGpuMem::TGpuMemUChar * MemSrc, TGpuMem::TGpuMemUChar * MemDst, TGpuMem::TGpuMemHalfFloat * MemU, TGpuMem::TGpuMemHalfFloat * MemV,bool Inverted);

	virtual ~TCVGeometry();

private:
	void * Gpu;

};

#include "TCV.h"
#include "TGpu.h"

#endif /* TCVGEOMETRY_H_ */
