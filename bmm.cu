//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "bmm.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// TILEX and TILEY are used to set the number of threads in a CUDA block 

#define TILEX 32
#define TILEY 16

#if (TILEX > TILEY)
#define TILEMAX TILEX
#define TILEMIN TILEY
#else
#define TILEMAX TILEY
#define TILEMIN TILEX
#endif // (TILEX > TILEY)


// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!

dim3 getDimGrid(const int m, const int n) {
	dim3 dimGrid(n/TILEX,n/TILEY);
	return dimGrid;
}
dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILEX,TILEY);
	return dimBlock;
}
__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n)
{

	// write your GPU kernel function here
	__shared__ float As[TILEY][TILEMAX]; // = A[bx*TILEY:(bx+1)*TILEY][:]
	__shared__ float Bs[TILEMAX][TILEX]; // = B[:][bx*TILEX:(bx+1)*TILEX]

	float temp = 0;
	for (int index = 0; index < (n/TILEMAX); index++)
	{
		for (int asIndex = 0; asIndex < (TILEMAX / TILEX); asIndex++)
		{
			As[ty][TILEX * asIndex + tx] = mem2d(ad, m, (by * TILEY + ty), (index * TILEMAX + asIndex * TILEX + tx));
		}

		for (int bsIndex = 0; bsIndex < (TILEMAX / TILEY); bsIndex++)
		{
		//int bsIndex = 1;

			Bs[TILEY * bsIndex + ty][tx] = mem2d(bd, m, (index * TILEMAX + bsIndex * TILEY + ty), (bx * TILEX + tx));
		}
		
		__syncthreads();
		for (int k = 0; k < TILEMAX; k++)
		{
			temp += As[ty][k] * Bs[k][tx];
		}
		__syncthreads();
	}
	mem2d(cd, m, (by * TILEY + ty), (bx * TILEX + tx)) = temp;
}
