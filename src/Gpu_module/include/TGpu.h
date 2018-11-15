/*
 * TGpu.h
 *
 *  Created on: 18/02/2015
 *      Author: pablo
 */

#ifndef TGPU_H_
#define TGPU_H_

#include "TCV.h"
#include "shared_EXPORTS.h"

class SHARED_EXPORT TGpu {
public:

    TGpu(int Block_X,int Block_Y);
    int CountGPUs();
	void PrintProperties(int Device_Num);
	void GetLastError();
	void SetDevice(int DevNum);
	void ResetDevice();
	int  iDivUp(int a, int b);
	void SetBlockSize(int Block_X,int Block_Y);
	int GetBlockX();
	int GetBlockY();

	void StartMeasurement();
	float StopMeasurement();
	
    ~TGpu();

    TCV * CV;

private:
	void InitTimer();
	void DestroyTimer();
    int BlockX;
    int BlockY;
	cudaEvent_t start, stop;
};

#endif /* TGPU_H_ */
