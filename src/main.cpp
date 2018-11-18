
//----------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <typeinfo>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//----------------------------------
#include "TGpu.h"

//----------------------------------
//using namespace cimg_library;
using namespace std;
using namespace cv;
//--------------------------------------------------------------------------
// MAIN
//--------------------------------------------------------------------------
int main(void)
{

	TGpu *myGpu = new TGpu(16,8);

	myGpu->SetCacheConfig(CacheConfig::PreferL1);

    cout << "Num of Gpus: "<<myGpu->CountGPUs() <<endl;
	myGpu->PrintProperties(0);

    Mat im_rgb_host = imread("C:\\Work\\GPU_CV\\data\\denso.png", CV_LOAD_IMAGE_COLOR);

    Mat im_gray_host(im_rgb_host.rows , im_rgb_host.cols, CV_8UC(1));
    //-----------------------------------------------------------------------------------------------------------------
    TGpuMem::TGpuMemUChar * im_rgb_device= new TGpuMem::TGpuMemUChar(myGpu, im_rgb_host.cols, im_rgb_host.rows,3);
	TGpuMem::TGpuMemUChar * im_gray_device = new TGpuMem::TGpuMemUChar(myGpu, im_rgb_host.cols, im_rgb_host.rows, 3);
    //-----------------------------------------------------------------------------------------------------------------

    float elapsed=0;

	myGpu->StartMeasurement();

	// Copy data from Host to Device
	im_rgb_device->CopyToDevice(im_rgb_host.data);

	// Convert from RGB to GRAY
	myGpu->CV->ColorSpace->RgbToGray(im_rgb_device, im_gray_device);

	// Copy data from Device to Host
	im_gray_host.data = im_gray_device->CopyFromDevice();

	elapsed = myGpu->StopMeasurement();

	cout << "Time (ms): " << elapsed  <<endl;

    //-----------------------------------------------------------------
	// Display image
	//-----------------------------------------------------------------
    namedWindow( "Display window 1", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window 1", im_gray_host);                   // Show our image inside it.
    moveWindow("Display window 1", 200, 200);

	waitKey(0);
    //-----------------------------------------------------------
    // Dispose
    //-----------------------------------------------------------
	im_rgb_host.release();
	im_gray_host.release();

    delete myGpu;
    delete im_rgb_device;
	delete im_gray_device;
   
	return 0;
}
