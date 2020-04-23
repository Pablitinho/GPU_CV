/*
 * TCV.cpp
 *
 *  Created on: 18/02/2015
 *      Author: pablo
 */

#include "TCV.h"
#include "CVCudaUtils.cuh"
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
//--------------------------------------------------------------------------
TCV::TCV(void * d_Gpu)
{
	ColorSpace = new TCVColorSpace(d_Gpu);
    Geometry   = new TCVGeometry(d_Gpu);
    Filters = new TCVFilters(d_Gpu);
    Features = new TCVFeatures(d_Gpu);
    Math = new TCVMath(d_Gpu);
    Motion = new TCVMotion(d_Gpu);
    Stereo =new TCVStereo(d_Gpu);
    Utils= new TCVUtils(d_Gpu);
}
//--------------------------------------------------------------------------
TCV::~TCV()
{
	delete ColorSpace;
	delete Geometry;
	delete Filters;
	delete Features;
	delete Math;
	delete Utils;
}
//--------------------------------------------------------------------------

