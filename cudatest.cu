//nvcc -arch sm_21 -o test -run --ptxas-options="-v" -lcufft cudatest.cu
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include </usr/include/cufft.h>

__global__ void cuda_print() {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;

  int bdx = blockDim.x;
  int bdy = blockDim.y;
  int bdz = blockDim.z;
  int gdx = gridDim.x;
  int gdy = gridDim.y;
  int gdz = gridDim.z;

  printf("Thread id (%d, %d, %d) \n", tx, ty,tz);
  //printf("blockdim (%d, %d, %d) \n", bdx, bdy, bdz);
  printf("Block idx (%d, %d, %d) \n", bx, by, bz);
  //printf("griddim  (%d, %d, %d) \n", gdx, gdy, gdz);
  printf("=======================\n");
}

int main(int argc, char** argv) {
  // do some cuda testing 
  cudaError_t res; 
  printf("entered \n");
  /*cuda_print<<<dim3(2,2,2), dim3(1,1,1)>>>();
  res = cudaDeviceSynchronize();*/
  /*cuda_print<<<dim3(1,1,1), dim3(2,2,2)>>>();
  res = cudaDeviceSynchronize();*/
  /*cuda_print<<<dim3(2,2), dim3(2,2)>>>();
  res = cudaDeviceSynchronize();
  cuda_print<<<dim3(2,2,1), dim3(1,2,3)>>>();
  res = cudaDeviceSynchronize();*/

  int n = 4;
  float* mat = (float*) malloc(n*n*n*sizeof(float));
  int i,j,k;
  for (i=0; i < n*n*n; i++) {
    *(mat+i) = i;//%n;
  }
  for (i=0; i<n; i++) {
    printf("======= x sheet %d =====\n", i);
    for (j=0; j<n; j++) {
      for (k=0; k<n; k++) {
        printf("%f ", *(mat+n*n*i+n*j+k));
      }
      printf("\n\n");
    }
  }

  
  float* gpu_in;
  cufftComplex* gpu_out;
  cudaMalloc(&gpu_in, n*n*n*sizeof(float));
  cudaMalloc(&gpu_out, n*n*(n/2+1)*sizeof(cufftComplex));
  cufftComplex* fft_out = (cufftComplex*) malloc(n*n*(n/2+1)*sizeof(cufftComplex));

  cudaMemcpy(gpu_in, mat, n*n*n*sizeof(float), cudaMemcpyHostToDevice);
  cufftHandle plan;
  cufftPlan3d(&plan, n,n,n, CUFFT_R2C);
  cufftExecR2C(plan, gpu_in, gpu_out);
  //cufftDestroy(plan);

  
  cudaMemcpy(fft_out, gpu_out, n*n*(n/2+1)*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  for (i=0; i<n; i++) {
    printf("======= x sheet %d =====\n", i);
    for (j=0; j<n; j++) {
      for (k=0; k<n/2+1; k++) {
        printf("%f ", ((cufftComplex)*(fft_out+n*n*i+n*j+k)).x);
        printf("+%fi ", ((cufftComplex)*(fft_out+n*n*i+n*j+k)).y);
      }
      printf("\n\n");
    }
  }

  cufftPlan3d(&plan, n,n,n, CUFFT_C2R);
  cufftExecC2R(plan, gpu_out, gpu_in);
  
  cudaMemcpy(mat, gpu_in, n*n*n*sizeof(float), cudaMemcpyDeviceToHost);
  for (i=0; i<n; i++) {
    printf("======= x sheet %d =====\n", i);
    for (j=0; j<n; j++) {
      for (k=0; k<n; k++) {
        printf("%f ", (*(mat+i*n*n+j*n+k))/(n*n*n));
      }
      printf("\n\n");
    }
  }

}
 
