/*
 * TCV.h
 *
 *  Created on: 18/02/2015
 *      Author: pablo
 */

#include "TCVColorSpace.h"
#include "TCVGeometry.h"
#include "TCVFilters.h"
#include "TCVFeatures.h"
#include "TCVMath.h"
#include "TCVMotion.h"
#include "TCVUtils.h"
#include "TCVStereo.h"

#ifndef TCV_H_
#define TCV_H_

//class TGpuMem;

#include "shared_EXPORTS.h"

class SHARED_EXPORT TCV
{

public:
	TCV(void * d_Gpu);
	virtual ~TCV();

	TCVColorSpace * ColorSpace;
	TCVGeometry * Geometry;
	TCVFilters * Filters;
    TCVFeatures * Features;
    TCVMath * Math;
    TCVMotion * Motion;
    TCVStereo * Stereo;
    TCVUtils * Utils;
};
//#include "TGpu.h"
//class TGpuMem;
//#include "TGpuMem.cu"

#endif /* TCV_H_ */
