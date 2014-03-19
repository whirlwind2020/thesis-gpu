#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <cufft.h>

/* Calculate the time derivative of fluid with velocity
 * ux, uy, uz. The size of any dimension of any cube
 * is size.  */
public float* dudt(float* ux, float* uy,
                          float* uz, int size) {
  cudaError_t lastError = cudaSuccess;
  // move ux to GPU
  void* ux_gpu;
  lastError = cudaMalloc(&ux_gpu, sizeof(ux));
  // gpuarrays++
  lastError = cudaMemcpy(ux_gpu, ux, 
                size*size*size, cudaMemcpyHostToDevice);

  // calculate fftderiv of ux
  
