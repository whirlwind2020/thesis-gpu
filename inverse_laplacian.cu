#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include </usr/include/cufft.h>

/* double-integrates in fourier-space by dividing
 * each spot by its distance^2 from zero. 
 * assumes it was invoked with size^2 blocks */
__global__ void integ_divide(cufftComplex* intermediate_gpu, int size) {
  int location = blockIdx.x*gridDim.x*gridDim.y;
  location += blockIdx.y*gridDim.y;
  
  float multiplier =(float)(blockIdx.x*blockIdx.x + blockIdx.y*blockIdx.y);
  printf("(%d %d %d) %f\n", blockIdx.x, blockIdx.y, blockIdx.z, multiplier);
  /*printf("Grid dim: dx %d dy. Thread idx x %d y %d\n\n",
      gridDim.x, gridDim.y, threadIdx.x, threadIdx.y);
  printf("idxX: %d, idxY: %d, dimX: %d dimY%d\n"
     " location %d multiplier %f\n\n",
      blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, location, multiplier);*/
  int z; 
  // don't divide by zero
  for (z = (multiplier ? 0 : 1); z < size/2+1; z++) { 
    float spot = -1.0/(multiplier+(float)(z*z));
    cufftComplex val = intermediate_gpu[location];
    val.x *= spot;
    val.y *= spot;
    intermediate_gpu[location] = val;
  }
}

/* take the inverse laplacian of input matrix lapl_p in GPU */
void inverse_laplacian(float* input, int size, float* output) {
  cufftResult res;
  cudaError_t cres;

  cufftComplex* intermediate_gpu;
  cres = cudaMalloc(&intermediate_gpu, size*size*(size/2+1)*sizeof(cufftComplex)); 
  printf("cudaMalloc %d\n", cres);

  /* take fftn(input)*/
  cufftHandle plan;
  res = cufftPlan3d(&plan, size, size, size, CUFFT_R2C);
  printf("plan %d\n", res);

  res = cufftExecR2C(plan, input, intermediate_gpu);
  printf("execd %d\n", res);
  
  res = cufftDestroy(plan);
  printf("destroy'd\n");

  /* divide index (x,y,z) by 1/(x^2+y^2+z^2)*/
  dim3 blocks(size,size, 1);
  dim3 threads(1,1,1);
  integ_divide<<<blocks, threads>>>(intermediate_gpu, size);
  cudaDeviceSynchronize();
  printf("done parallel\n");
  // but zero out the first index.
  cudaMemset(intermediate_gpu, 0, 1*sizeof(cufftComplex));

  printf("complete threads\n");

  /* now inverse transform */
  res = cufftPlan3d(&plan, size, size, size, CUFFT_C2R);
  printf("plan iff %d\n", res);

  res = cufftExecC2R(plan, intermediate_gpu, output);
  printf("plan iff %d\n", res);

  res = cufftDestroy(plan);
  printf("done w stuff\n");
}


int main(int argc, char** argv) {
  printf("ya ya\n");

  int n = 4;
  float* input = (float*) malloc(n*n*n*sizeof(float));
  float* output =(float*) malloc(n*n*n*sizeof(float));

  int i,j,k;
  for (i=0; i<n*n*n; i++) {
    input[i] = i;
  }

  float* gpu_input;
  float* gpu_output;
  cudaError_t res;
  res = cudaMalloc((void**)&gpu_input, n*n*n*sizeof(float));

  res = cudaMalloc((void**)&gpu_output, n*n*n*sizeof(float));

  res = cudaMemcpy(gpu_input, input, n*n*n*sizeof(float), cudaMemcpyHostToDevice);

  inverse_laplacian(gpu_input, n, gpu_output);
  
  res = cudaMemcpy(output, gpu_output, n*n*n*sizeof(float), cudaMemcpyDeviceToHost);
  
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      for (k=0; k<n; k++) {
        printf("%3.2F ", *(output+i*n*n+j*n+k));
      }
      printf("\n\n");
    }
    printf("======= x sheet %d ======= \n", i);
  }


  free(input); free(output);
  cudaFree(gpu_input); cudaFree(gpu_output);
}
