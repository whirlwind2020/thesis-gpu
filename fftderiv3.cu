// nvcc -arch sm_21 -o fftderiv3 -run --ptxas-options="-v" -lcufft fftderiv3.cu
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include </usr/include/cufft.h>

/* take derivative in fourier-space by multiplying 
 * axis-index by that index times imaginary unit.
 * Assumes size^2 blocks, each of which goes along a z-row bc caching*/
__global__ void deriv_multiply(cufftComplex* intermediate_gpu, int size, int axis) {
  float totaldim = (float)size; // scaling factor to normalize
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
    int idx = blockIdx.x*size*(size/2+1) + blockIdx.y*size;
    // multiply all these by y
    for (z = 0; z < size; z++) {
      oldval = intermediate_gpu[idx+z];
      /* by imaginary unit times y */
      newval.y = oldval.x*blockIdx.y;
      newval.x = oldval.y*blockIdx.y;
      
      // now scale down
      newval.x /= totaldim;
      newval.y /= totaldim;

      intermediate_gpu[idx+z] = newval;
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

void fftderiv3byslice(float* input, int size, float* output, int axis) {
  cufftResult res;
  cudaError_t cres;

  cufftComplex* intermediate_gpu;
  cres = cudaMalloc(&intermediate_gpu, size*size*(size/2+1)*sizeof(cufftComplex)); // expensive
  cufftComplex* igpu_host = (cufftComplex*)malloc(size*size*(size/2+1)*sizeof(cufftComplex));
  printf("malloc %d\n", cres);
  
  cufftHandle plan;
  //below works for axis==2!
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
        intermediate_gpu+size*(size/2+1)*x);
    cudaMemcpy(igpu_host, intermediate_gpu+size*(size/2+1)*x, size*(size/2+1)*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    printf("ysheet %d: %d\n", x, res);
    for (int j=0; j<size/2+1; j++) {
      for (int k=0; k<size; k++) {
        printf("%f ", ((cufftComplex)*(igpu_host+size*j+k)).x);
        printf("+%fi ", ((cufftComplex)*(igpu_host+size*j+k)).y);
      }
      printf("\n\n");
    } 
  }
  printf("\n\n");
  // cleanup
  cufftDestroy(plan);
  free(igpu_host);
  


  /* for testing, copy everything over at once and make sure it makes sense  */
  cufftComplex* fullstuff = (cufftComplex*) malloc(size*size*(size/2+1)*sizeof(cufftComplex));
  cudaMemcpy(fullstuff, intermediate_gpu, size*size*(size/2+1)*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  int i,j,k;
  for(i=0;i<size;i++) {
    for(j=0;j<size/2+1;j++) {
      for(k=0;k<size;k++) {
        int idx = i*size*(size/2+1)+j*size+k;
        cufftComplex num = (cufftComplex)*(fullstuff+idx);
        printf("%f +%fi ", num.x, num.y);
      }
      printf("\n\n");
    }
    printf("==== sheet %d ====\n", i);
  }
  /* end test move */

  /* multiply to take deriv in fourier-space */
  dim3 blocks(size, size/2+1);
  deriv_multiply<<<blocks, 1>>>(intermediate_gpu, size, axis);
  printf("finished kernel multiplication\n \n ");

  /* confirm stuff still looks reasonabl3e */
  cudaMemcpy(fullstuff, intermediate_gpu, size*size*(size/2+1)*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  for(i=0;i<size;i++) {
    for(j=0;j<size/2+1;j++) {
      for(k=0;k<size;k++) {
        int idx = i*size*(size/2+1)+j*size+k;
        cufftComplex num = (cufftComplex)*(fullstuff+idx);
        printf("%f +%fi ", num.x, num.y);
      }
      printf("\n\n");
    }
    printf("==== sheet %d ====\n", i);
  }

  /* inverse transform */
  i_stride = size, o_stride = size;
  i_dist = 1, o_dist = 1;
  i_nembed = (size/2+1)*size, o_nembed = size*size;

  res = cufftPlanMany(&plan, 1, &size, &i_nembed,
      i_stride, i_dist, &o_nembed, o_stride, o_dist,
      CUFFT_C2R, size);
  printf("yplaninverse %d\n", res);

  float* holder = (float*) malloc(size*size*sizeof(float));
  for (x = 0; x < size; x++) {
    res = cufftExecC2R(plan, intermediate_gpu+(size/2+1)*size*x,
        output+size*size*x);
    cudaMemcpy(holder, output+size*size*x, size*size*sizeof(float), cudaMemcpyDeviceToHost);
    printf("yshet %d\n", x);
    for (int j=0; j<size;j++) {
      for (int k=0; k<size;k++) {
        int idx = j*size+k;
        printf("%f ", (float)*(holder+idx));
      }
      printf("\n\n");
    }
  }

  free(holder);
  cudaFree(intermediate_gpu);
  cufftDestroy(plan);
    

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
  //int n_freqs = size/2+1;
  int n_freqs = size; //n-point to n-point transform
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
        int val = n*n*i+n*j+k;
        *(input+i*n*n+j*n+k) = val;
      }
    }
  }
  printf("done making initial mat\n");

  float* gpu_input;
  float* gpu_output;
  cudaError_t res;
  res = cudaMalloc((void**)&gpu_input, n*n*n*sizeof(float));
  printf("input malloc res %d\n", res);
  res = cudaMalloc((void**)&gpu_output, n*n*n*sizeof(float));
  printf("output malloc res %d\n", res);
  res = cudaMemcpy(gpu_input, input, n*n*n*sizeof(float), cudaMemcpyHostToDevice);
  
//  fftderiv3(gpu_input, n, gpu_output, 2);
  fftderiv3byslice(gpu_input, n, gpu_output, 2);

  res = cudaMemcpy(output, gpu_output, n*n*n*sizeof(float), cudaMemcpyDeviceToHost);
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
        printf("%f ", *(output+i*n*n+j*n+k)); 
        //printf("%d ", abs(*(output+i*n*n+j*n+k) - *(input+i*n*n+j*n+k)) < .3);
      }
      printf("\n\n");
    }
    printf("======= x sheet %d ==== \n", i);
  }

  free(input); free(output);
  cudaFree(gpu_input); cudaFree(gpu_output);
}
