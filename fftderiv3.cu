#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include </usr/include/cufft.h>

/* take derivative in fourier-space by multiplying 
 * axis-index by that index times imaginary unit.
 * Assumes size^2 blocks, each of which goes along a z-row bc caching*/
__global__ void deriv_multiply(cufftComplex* intermediate_gpu, int size, int axis) {
  int location = blockIdx.x*blockDim.x*blockDim.y;
  location += blockIdx.y*blockDim.y;
  if (location > size*size*(size/2+1)-(size/2+1)) {
    // since z will range from 0 to size/2
    // TODO: fail
    return;
  }
  if (axis < 1 || axis > 3) {
    // TODO: fail
    return;
  }

  cufftComplex oldval;
  cufftComplex newval;
  int z;
  if (axis == 1) {
    // multiply all these by x
    for (z = 0; z < size/2+1; z++) {
      oldval = intermediate_gpu[location+z];
      newval.y = oldval.x*blockIdx.x;
      newval.x = oldval.y*-1*blockIdx.x;
      intermediate_gpu[location+z] = newval;
    }

  } else if (axis == 2) {
    // multiply all these by y
    for (z = 0; z < size/2+1; z++) {
      oldval = intermediate_gpu[location+z];
      newval.y = oldval.x*blockIdx.y;
      newval.x = oldval.y*-1*blockIdx.y;
      intermediate_gpu[location+z] = newval;
    }

  } else if (axis == 3) {
    // multiply index z by z
    for (z = 0; z < size/2+1; z++) {
      oldval = intermediate_gpu[location+z];
      newval.y = oldval.x*z;
      newval.x = oldval.y*-1*z;
      intermediate_gpu[location+z] = newval;
    }

  } 
}

/* takes the derivative along specified axis of 
 * input, and puts the result in output.
 * input and output are on the GPU. Size is cube-side-length (not volume)*/
void fftderiv3(float* input, int size, float* output, int axis) {
  if (axis < 1 || axis > 3) {
    //TODO: fail hard
  } 
  cufftResult res;
  cudaError_t cres;

  cufftComplex* intermediate_gpu;
  cres = cudaMalloc(&intermediate_gpu, size*size*(size/2+1)*sizeof(cufftComplex)); // expensive
  printf("malloc %d\n", cres);
  
  cufftHandle plan;

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

    printf("plan %d\n", res);
    res = cufftExecR2C(plan, input, intermediate_gpu);
    printf("exec %d\n", res);

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
    printf("yplan %d\n", res);

    int x;
    for (x = 0; x < size; x++) {
      res = cufftExecR2C(plan, input+size*size*x,
                               intermediate_gpu+size*x);
      printf("ysheet %d: %d\n", x, res);
    }
  }

  res = cufftDestroy(plan);
  printf("finished ffts\n");

  /* FOR TESTING: just send back the FFT */ 
  /*cres = cudaMemcpy(output, intermediate_gpu, size*size*size, cudaMemcpyDeviceToDevice);
  return;*/
  /* END TESTING */

  // intermediate_gpu is now populated with FFT, deriv
  // TESTING REMOVED before
  dim3 blocks(size, size);
  deriv_multiply<<<blocks, 1>>>(intermediate_gpu, size, axis);
  printf("finished kernelmult\n");

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
    int i_stride = size, o_stride = size;
    int i_dist = 1, o_dist = 1;
    int i_nembed = size*(size/2+1), o_nembed = size*size;

    res = cufftPlanMany(&plan, 1, &n_freqs, &i_nembed,
        i_stride, i_dist, &o_nembed, o_stride, o_dist,
        CUFFT_C2R, size);

    int x;
    for (x = 0; x < size; x++) {
      res = cufftExecC2R(plan, intermediate_gpu+size*size*x, 
                               output+size*size*x);
    }
  }

  res = cufftDestroy(plan);
  cudaFree(intermediate_gpu);

  printf("totes done\n");
}


int main(int argc, char** argv) {
  printf("well, we're here\n.");

  int n = 4;
  float* input = (float*)malloc(n*n*n*sizeof(float));
  float* output = (float*)malloc(n*n*n*sizeof(float));

  float PI = 3.14159; // for testing 
  int i, j, k;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
        //*(input+i*n*n+j*n+k) = (float) rand()/ 15.0;
        //*(input+i*n*n+j*n+k) = (float)sin(5*k*PI/180);
        *(input+i*n*n+j*n+k) = j;
      }
    }
  }
  printf("done making random mat\n");

  float* gpu_input;
  float* gpu_output;
  cudaError_t res;
  res = cudaMalloc((void**)&gpu_input, n*n*n*sizeof(float));
  printf("input malloc res %d\n", res);
  res = cudaMalloc((void**)&gpu_output, n*n*n*sizeof(float));
  printf("output malloc res %d\n", res);
  res = cudaMemcpy(gpu_input, input, n*n*n*sizeof(float), cudaMemcpyHostToDevice);
  
  fftderiv3(gpu_input, n, gpu_output, 2);

  res = cudaMemcpy(output, gpu_output, n*n*n*sizeof(float), cudaMemcpyDeviceToHost);
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
        printf("%3.2F ", *(output+i*n*n+j*n+k)); 
        //printf("%d ", abs(*(output+i*n*n+j*n+k) - *(input+i*n*n+j*n+k)) < .3);
      }
      printf("\n\n");
    }
    printf("======= x sheet %d ==== \n", i);
  }

  free(input); free(output);
  cudaFree(gpu_input); cudaFree(gpu_output);
}
