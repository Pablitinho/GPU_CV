/*
 * TColorSpace.h
 *
 *  Created on: 18/02/2015
 *      Author: pablo
 */

#ifndef TCVCOLORSPACE_H_
#define TCVCOLORSPACE_H_

//class TGpuMem;
//class TGpu;
//class TCV;

#include "TGpuMem.h"
//--------------------------------------------------------------------------
class SHARED_EXPORT TCVColorSpace {

public:
	TCVColorSpace(void * d_Gpu);
	void RgbToGray(TGpuMem::TGpuMemUChar  * RGB_Image, TGpuMem::TGpuMemUChar * Gray_Image);
	void RgbToGray(TGpuMem::TGpuMemUChar  * RGB_Image, TGpuMem::TGpuMemHalfFloat * Gray_Image);
	virtual ~TCVColorSpace();

private:
	void * Gpu;

};
#include "TCV.h"
#include "TGpu.h"

#endif /* TCVCOLORSPACE_H_ */

//#include "TColorSpace.cpp"

