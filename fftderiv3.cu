#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <cufft.h>

/* takes the derivative along specified axis of 
 * input, and puts the result in output.
 * input and output are on the GPU*/
public void fftderiv3(float* input, int size, float* output, int axis) {
  // general plan: transform a row at a time.
  // this means move rows around to be compatible and then
  // move them back later (w memcpy) into output. Then parallel multiply i
  // cuComplex has c.r and c.i for real and imaginary
  
  cufftComplex* intermediate_gpu;
  cudaMalloc(&intermediate_gpu, size*size*(size/2+1)); //'spensive

  if (axis == 1) {
    // for each pair j,k, perform an fft along [][j][k]
  } else if (axis == 2) {

  } else if (axis == 3) {
    // for each pair i,j perform fft along [i][j]
    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_R2C, size*size); //batch size^2 for each col

    int complexsize = size/2 + 1;
    
    float* thisrow_gpu;
    cufftComplex* thisrowxform_gpu;
    cudaMalloc(&thisrow_gpu, size);
    cudaMalloc(&thisrowxform_gpu, complexsize*sizeof(cufftComplex));
    
    // do all n ffts
    cudaExecR2C(plan, input, intermediate_gpu, CUFFT_FORWARD);
    int i, j;
    for (i = 0; i < size; i++) {
      for (j = 0; j < size; j++) {
        cudaMemcpy(thisrow_gpu, input + size*size*i + size*j, 
                    size*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaExecR2C(plan, thisrow_gpu, thisrowxform_gpu, CUFFT_FORWARD);
        // now move it to our 3d fft box
        cudaMemcpy(intermediate_gpu+size*size*i+size*j, thisrowxform_gpu,
                    complexsize*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
      }
    }

    // free resources
    cufftDestroy(plan);


    // intermediate_gpu contains fft(input, axis)
    // parallel multiply index j by ij (imaginary unit)

    // this dimension assumes z direction
    dim3 blocks(size, size, size/2 + 1);
    deriv_multiplier<<<blocks, 1>>>(intermediate, size, axis);


    // now inverse fft
    
      
    
  }
}

public void fftderiv3general(float* input, int size, int axis) { 
  cufftComplex* intermediate_gpu;
  cudaMalloc(&intermediate_gpu, size*size*(size/2+1)); //'spensive

}




/* method to multiply each element of the matrix by 
 * its index (along axis) and the imaginary unit i.*/
__global__ void deriv_multiplier(cufftComplex* intermediate, int size, int axis) {
 // so far still assuming in the z direction. double-check later 
  int location = blockIdx.x*blockDim.x*blockDim.y;
  location += blockIdx.y*blockDim.y;
  location += blockIdx.z;
  if (location >= size*size*size) {
    /* shouldn't happen by blockDim specification*/
    return;
  }

  int location_mult;
  if (axis == 1)
    location_mult = blockIdx.x;
  else if (axis == 2)
    location_mult = blockIdx.y;
  else if (axis == 3)
    location_mult = blockIdx.z;
  
  cufftComplex oldval = intermediate[location];
  cufftComplex newval;
  newval.r = oldval.c*-1*location_mult;
  newval.c = oldval.r*location_mult;

  intermediate[location] = newval;
}
