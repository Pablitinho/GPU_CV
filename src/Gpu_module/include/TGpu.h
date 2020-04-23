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

enum CacheConfig { DefaultCache=0, PreferShared=1, PreferL1=2, Equal=3};

class SHARED_EXPORT TGpu {
public:

    TGpu(int Block_X,int Block_Y,int Device_Num);
    int CountGPUs();
	void PrintProperties(int Device_Num);
	void GetLastError();
	void SetDevice(int DevNum);
	int GetDevice();
	void ResetDevice();
	int  iDivUp(int a, int b);
	void SetBlockSize(int Block_X,int Block_Y);
	int GetBlockX();
	int GetBlockY();

	void StartMeasurement();
	float StopMeasurement();
	
	void SetCacheConfig(CacheConfig config);
	void GetMemoryInfo(size_t * FreeMemory,size_t *TotalMemory);
    ~TGpu();

    TCV * CV;

private:
	
	bool Allow_zero_copy();
	void InitTimer();
	void DestroyTimer();
    int BlockX;
    int BlockY;
	cudaEvent_t start, stop;
	bool support_zero_copy;
};

#endif /* TGPU_H_ */
