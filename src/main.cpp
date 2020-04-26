
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

using namespace std;
using namespace cv;


//--------------------------------------------------------------------------
// MAIN
//--------------------------------------------------------------------------
void imageResize(TGpu *myGpu)
{
	float factor = 2.5;

	Mat im_rgb_host = imread("..\\data\\denso.png", IMREAD_UNCHANGED);

	Mat im_gray_host(im_rgb_host.rows, im_rgb_host.cols, CV_8UC(1));
	Mat im_gray_host_resized(im_rgb_host.rows*factor, im_rgb_host.cols*factor, CV_8UC(1));
	//------------------------------------------------------------
	//-----------------------------------------------------------------------------------------------------------------
	TGpuMem::TGpuMemUChar * im_rgb_device = new TGpuMem::TGpuMemUChar(myGpu, im_rgb_host.cols, im_rgb_host.rows, 3, false);
	TGpuMem::TGpuMemUChar * im_gray_device = new TGpuMem::TGpuMemUChar(myGpu, im_rgb_host.cols, im_rgb_host.rows, 1, false);
	TGpuMem::TGpuMemUChar * im_gray_device_resized = new TGpuMem::TGpuMemUChar(myGpu, im_rgb_host.cols*factor, im_rgb_host.rows*factor, 1, false);
	//-----------------------------------------------------------------------------------------------------------------
	float elapsed = 0;

	myGpu->StartMeasurement();

	// Copy data from Host to Device
	im_rgb_device->CopyToDevice(im_rgb_host.data);

	// Convert from RGB to GRAY
	myGpu->CV->ColorSpace->RgbToGray(im_rgb_device, im_gray_device);
	// Resize image
	//myGpu->CV->Geometry->ResizeBilinear(im_gray_device, im_gray_device_resized);
	myGpu->CV->Geometry->Resize(im_gray_device, im_gray_device_resized);

	// Copy data from Device to Host
	im_gray_host.data = im_gray_device->CopyFromDevice();
	im_gray_host_resized.data = im_gray_device_resized->CopyFromDevice();

	elapsed = myGpu->StopMeasurement();

	cout << "Time (ms): " << elapsed << endl;

	//-----------------------------------------------------------------
	// Display image
	//-----------------------------------------------------------------
	namedWindow("Image Gray", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Image Gray", im_gray_host_resized);                   // Show our image inside it.
	moveWindow("Image Gray", 200, 200);

	waitKey(0);
	//-----------------------------------------------------------
	// Dispose
	//-----------------------------------------------------------
	im_rgb_host.release();
	im_gray_host.release();
	delete im_rgb_device;
	delete im_gray_device;
}
//-------------------------------------------------------------------------------------------------------------------
void opticalFlow(TGpu *myGpu,Mat im_rgb_host_0,Mat im_rgb_host_1)
{

	Mat opticalFlowMat(im_rgb_host_0.rows, im_rgb_host_0.cols, CV_8UC(3));

	TGpuMem::TGpuMemUChar * im_gray_device = new TGpuMem::TGpuMemUChar(myGpu, im_rgb_host_1.cols, im_rgb_host_1.rows, 1, false);

	TGpuMem::TGpuMemUChar * im_rgb_device_0 = new TGpuMem::TGpuMemUChar(myGpu, im_rgb_host_0.cols, im_rgb_host_0.rows, 3, false);
	TGpuMem::TGpuMemUChar * im_rgb_device_1 = new TGpuMem::TGpuMemUChar(myGpu, im_rgb_host_1.cols, im_rgb_host_1.rows, 3, false);
	TGpuMem::TGpuMemUChar * opticalFlow = new TGpuMem::TGpuMemUChar(myGpu, im_rgb_host_1.cols, im_rgb_host_1.rows, 3, false);


	TGpuMem::TGpuMemHalfFloat * U = new TGpuMem::TGpuMemHalfFloat(myGpu, im_rgb_host_0.cols, im_rgb_host_0.rows, 1, false);
	TGpuMem::TGpuMemHalfFloat * V = new TGpuMem::TGpuMemHalfFloat(myGpu, im_rgb_host_1.cols, im_rgb_host_1.rows, 1, false);

	TGpuMem::TGpuMemHalfFloat * im_gray_hf_device_0 = new TGpuMem::TGpuMemHalfFloat(myGpu, im_rgb_host_0.cols, im_rgb_host_0.rows, 1, false);
	TGpuMem::TGpuMemHalfFloat * im_gray_hf_device_1 = new TGpuMem::TGpuMemHalfFloat(myGpu, im_rgb_host_1.cols, im_rgb_host_1.rows, 1, false);

	// Copy to gpu
	im_rgb_device_0->CopyToDevice(im_rgb_host_0.data);
	im_rgb_device_1->CopyToDevice(im_rgb_host_1.data);

	// RGB-> Gray
	myGpu->CV->ColorSpace->RgbToGray(im_rgb_device_0, im_gray_hf_device_0);
	myGpu->CV->ColorSpace->RgbToGray(im_rgb_device_1, im_gray_hf_device_1);

	// Create pyramid
	myGpu->CV->Motion->InitPyramid(10, 20, 0.5, im_rgb_host_0.cols, im_rgb_host_0.rows);
	// Comput Optical flow
	myGpu->CV->Motion->AniTVL1(im_gray_hf_device_0, im_gray_hf_device_1, U, V, 150, 2, 0.7, 0.6, false);
	// Convert optical flow to color
	myGpu->CV->Utils->OpticalFlowToColor(U, V, opticalFlow, 0.5);
	// Copy from device
	opticalFlowMat.data = opticalFlow->CopyFromDevice();
	// Show optical flow
	namedWindow("Optical Flow", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Optical Flow", opticalFlowMat);                   // Show our image inside it.
	moveWindow("Optical Flow", 200, 200);
	waitKey(0);
}

void stereo(TGpu *myGpu,Mat im_rgb_host_0, Mat im_rgb_host_1)
{

	Mat disparity_host(im_rgb_host_0.rows, im_rgb_host_0.cols, CV_8UC(1));

	TGpuMem::TGpuMemUChar * im_rgb_device_0 = new TGpuMem::TGpuMemUChar(myGpu, im_rgb_host_0.cols, im_rgb_host_0.rows, 3, false);
	TGpuMem::TGpuMemUChar * im_rgb_device_1 = new TGpuMem::TGpuMemUChar(myGpu, im_rgb_host_1.cols, im_rgb_host_1.rows, 3, false);

	TGpuMem::TGpuMemHalfFloat * disparity_device_hf = new TGpuMem::TGpuMemHalfFloat(myGpu, im_rgb_host_1.cols, im_rgb_host_1.rows, 1, false);
	TGpuMem::TGpuMemUChar * disparity_device_char = new TGpuMem::TGpuMemUChar(myGpu, im_rgb_host_1.cols, im_rgb_host_1.rows, 1, false);

	TGpuMem::TGpuMemHalfFloat * im_gray_hf_device_0 = new TGpuMem::TGpuMemHalfFloat(myGpu, im_rgb_host_0.cols, im_rgb_host_0.rows, 1, false);
	TGpuMem::TGpuMemHalfFloat * im_gray_hf_device_1 = new TGpuMem::TGpuMemHalfFloat(myGpu, im_rgb_host_1.cols, im_rgb_host_1.rows, 1, false);

	// Copy to gpu
	im_rgb_device_0->CopyToDevice(im_rgb_host_0.data);
	im_rgb_device_1->CopyToDevice(im_rgb_host_1.data);

	// RGB-> Gray
	myGpu->CV->ColorSpace->RgbToGray(im_rgb_device_0, im_gray_hf_device_0);
	myGpu->CV->ColorSpace->RgbToGray(im_rgb_device_1, im_gray_hf_device_1);
	
	// Create pyramid
	myGpu->CV->Stereo->InitPyramid(16, 20, 0.5, im_rgb_host_0.cols, im_rgb_host_0.rows);
	// Comput Disparity
	myGpu->CV->Stereo->AniTVL1_Stereo(im_gray_hf_device_0, im_gray_hf_device_1, disparity_device_hf, 250,8, 0.7, 0.7, true);

	// Convert Disparity to color
	myGpu->CV->Utils->StereoToColor(disparity_device_hf, disparity_device_char);

	// Copy from device
	disparity_host.data = disparity_device_char->CopyFromDevice();
	// Show optical flow
	namedWindow("Stereo", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Stereo", disparity_host);                   // Show our image inside it.
	moveWindow("Stereo", 200, 200);
	waitKey(0);
}
int main(void)
{
	TGpu *myGpu = new TGpu(16,8,0);

	myGpu->SetCacheConfig(CacheConfig::PreferL1);

    cout << "Num of Gpus: "<<myGpu->CountGPUs() <<endl;
	cout << "Device ID: " << myGpu->GetDevice() << endl;
	myGpu->PrintProperties(0);

	Mat im_rgb_host_opticalflow_0 = imread("..\\data\\Hydrangea\\frame10.png", IMREAD_UNCHANGED);
	Mat im_rgb_host_opticalflow_1 = imread("..\\data\\Hydrangea\\frame11.png", IMREAD_UNCHANGED);

	opticalFlow(myGpu, im_rgb_host_opticalflow_0, im_rgb_host_opticalflow_1);

	Mat im_rgb_host_stereo_0 = imread("..\\data\\piano\\frame8.png", IMREAD_UNCHANGED);
	Mat im_rgb_host_stereo_1 = imread("..\\data\\piano\\frame9.png", IMREAD_UNCHANGED);
	stereo(myGpu, im_rgb_host_stereo_0, im_rgb_host_stereo_1);

	delete myGpu;

	return 0;
}
