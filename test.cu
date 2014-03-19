// nvcc -arch sm_21 -o test -run --keep --ptxas-options="-v" test.cu
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void transpose (int* Input, int* Output) { 
}
