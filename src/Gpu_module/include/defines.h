#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>

//--------------------------------------------------------------------------
// Macro to check Errors
//--------------------------------------------------------------------------
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define M_PI 3.14159f