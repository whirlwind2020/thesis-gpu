#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <cufft.h>

/* takes the derivative along specified axis of 
 * input, and puts the result in output.
 * input and output are on the GPU*/
public void fftderiv3(float* input, int size, float* output, int axis) {
  if (axis < 1 || axis > 3) {
    //TODO: fail hard
  } 

  cufftComplex* intermediate_gpu;
  cudaMalloc(&intermediate_gpu, size*size*(size/2+1)); // expensive
  
  cufftHandle plan;
  cufftResult res;

  if (axis != 2) {
    int i_stride, i_dist;
    int o_stride, o_dist;
    int i_nembed = size*size*size , o_nembed = size*size*(size/2+1);
    int batch = size*size; // number of rows to be FFT'd

    if (axis == 1) {
      /* [1][0][0] to [0][0][0] is n^2, but they are
       * adjacent in signal-space. */
      i_stride = o_stride = size*size;
      i_dist = o_dist = 1;

    } else if (axis == 3) {
      /* simple case. we have size^2 consecutive
       * signals of length size */ 
      i_stride = o_stride = 1;
      i_dist = o_dist = size;
    }

    /* now that we've set our stride and between-signals distance*/
    res = cufftPlanMany(&plan, 1, &size, &i_nembed, 
        i_stride, i_dist, &o_nembed, o_stride, o_dist,
        CUFFT_R2C, batch);
    res = cufftExecR2C(plan, input, intermediate_gpu);

  } else { /* axis == 2*/
    /* more complicated for Y. 
     * stride b/n consecutive signal values is size
     * for fixed x, signals start at 0:n-1
     * but incraesed x repeats at size*size (loop over x) */
    int i_stride = size, o_stride = size;
    int i_dist = 1, o_dist = 1;
    int i_nembed = size*size, o_nembed = size*(size/2+1);

    res = cufftPlanMany(&plan, 1, &size, &i_nembed,
        i_stride, i_dist, &o_nembed, o_stride, o_dist,
        CUFFT_R2C, size);

    int x;
    for (x = 0; x < size; x++) {
      res = cufftExecR2C(plan, input+size*x, intermediate_gpu+size*x);
      //TODO: wrong. adjust for size/2+1. also is it +size*size*x?
    }
  }

  res = cufftDestroy(plan);

  // intermediate_gpu is now populated with FFT, deriv
  dim3 blocks(size, size);
  deriv_multiply<<<blocks, 1>>>(intermediate_gpu, size, axis);

  // now i fft
  int n_freqs = size/2+1;
  if (axis != 2) {
    int i_stride, i_dist;
    int o_stride, o_dist;
    int i_nembed = size*size*(size/2+1), o_nembed = size*size*size;
    int batch = size*size; 

    if (axis == 1) {
      i_stride = o_stride = size*size;
      i_dist = o_dist = 1;

    } else if (axis == 3) {
      /* simple case */
      i_stride = o_stride = 1;
      i_dist = o_dist = size;
    }
    

    res = cufftPlanMany(&plan, 1, &n_freqs, &i_nembed,
        i_stride, i_dist, &o_nembed, o_stride, o_dist,
        CUFFT_C2R, batch);
    res = cufftExecC2R(plan, intermediate_gpu, output); 
  } else { /* axis == 2 */
    int i_stride = size; o_stride = size;
    int i_dist = 1, o_dist = 1;
    int i_nembed = size*(size/2+1), o_nembed = size*size;

    res = cufftPlanMany(&plan, 1, &n_freqs, &i_nembed,
        i_stride, i_dist, &o_nembed, o_stride, o_dist,
        CUFFT_C2R, size);

    int x;
    for (x = 0; x < size; x++) {
      res = cufftExecC2R(plan, intermediate_gpu+size*x, output+size*x);
      //TODO: adjust for size/2+1. also adjust for size*size*x?
    }
  }

  res = cufftDestroy(plan);
  cudaFree(intermediate_gpu);
}

/* take derivative in fourier-space by multiplying 
 * axis-index by that index times imaginary unit.
 * Assumes size^2 blocks, each of which goes along a z-row bc caching*/
__global__ void deriv_multiply(cufftComplex* intermediate_gpu, int size, int axis) {
  int location = blockIdx.x*blockDim.x*blockDim.y;
  location += blockIdx.y*blockDim.y;
  if (location >= size*size*size) {
    // TODO: fail
    return;
  }
  if (axis < 1 || axis > 3) {
    // TODO: fail
    return;
  }

  cufftComplex oldval = intermediate_gpu[location];
  cufftComplex newval;
  int z;
  if (axis == 1) {
    // multiply all these by x
    for (z = 0; z < size; z++) {
      newval.c = oldval.r*blockIdx.x;
      newval.r = oldval.c*-1*blockIdx.x;
      intermediate_gpu[location] = newval;
    }

  } else if (axis == 2) {
    // multiply all these by y
    for (z = 0; z < size; z++) {
      newval.c = oldval.r*blockIdx.y;
      newval.r = oldval.c*-1*blockIdx.y;
      intermediate_gpu[location] = newval;
    }

  } else if (axis == 3) {
    // multiply index z by z
    for (z = 0; z < size; z++) {
      newval.c = oldval.r*z;
      newval.r = oldval.c*-1*z;
      intermediate_gpu[location] = newval;
    }

  } 
}

