//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>
//#include <cuda_runtime.h>
//#include <cublas_v2.h>
//#include<helper_cuda.h>
//#include<device_launch_parameters.h>
//
//
//__host__ __device__ unsigned int IDX(unsigned int i, unsigned  int j, unsigned int ld) { return j * ld + i; }
//
//#define PERR(call) \
//  if (call) {\
//   fprintf(stderr, "%s:%d Error [%s] on "#call"\n", __FILE__, __LINE__,\
//      cudaGetErrorString(cudaGetLastError()));\
//   exit(1);\
//  }
//#define ERRCHECK \
//  if (cudaPeekAtLastError()) { \
//    fprintf(stderr, "%s:%d Error [%s]\n", __FILE__, __LINE__,\
//       cudaGetErrorString(cudaGetLastError()));\
//    exit(1);\
//  }
//
//__device__ float det_kernel(float* a_copy, unsigned int* n, cublasHandle_t* hdl) 
//{
//    int* info = (int*)malloc(sizeof(int)); info[0] = 0;
//    int batch = 1; int* p = (int*)malloc(*n * sizeof(int));
//    float** a = (float**)malloc(sizeof(float*));
//    *a = a_copy;
//    cublasSgetrfBatched(*hdl, *n, a, *n, p, info, batch);
//    cudaDeviceSynchronize();
//    unsigned int i1;
//    float res = 1;
//    for (i1 = 0; i1 < (*n); ++i1)res *= a_copy[IDX(i1, i1, *n)];
//    return res;
//}
//
//__global__ void runtest(float* a_i, unsigned int n) {
//    cublasHandle_t hdl; cublasCreate_v2(&hdl);
//    printf("det on GPU:%f\n", det_kernel(a_i, &n, &hdl));
//    cublasDestroy_v2(hdl);
//}
//
//int main()
//{
//    float a[] = {
//      1,   2,   3,
//      0,   4,   5,
//      1,   0,   0 };
//    //cudaSetDevice(1);//GTX780Ti on my machine,0 for GTX1080
//    unsigned int n = 3, nn = n * n;
//    printf("a is \n");
//    for (int i = 0; i < n; ++i) 
//    {
//        for (int j = 0; j < n; j++) 
//            printf("%f, ", a[IDX(i, j, n)]);
//        printf("\n");
//    }
//
//    float* a_d;
//    PERR(cudaMalloc((void**)&a_d, nn * sizeof(float)));
//    PERR(cudaMemcpy(a_d, a, nn * sizeof(float), cudaMemcpyHostToDevice));
//
//    dim3 block(128, 1);
//    dim3 cuda_grid_size = dim3((1 + block.x - 1) / block.x, 1);
//    runtest << <cuda_grid_size, block >> > (a_d, n);
//    cudaDeviceSynchronize();
//    ERRCHECK;
//
//    PERR(cudaMemcpy(a, a_d, nn * sizeof(float), cudaMemcpyDeviceToHost));
//    float res = 1;
//    for (int i = 0; i < n; ++i)
//        res *= a[IDX(i, i, n)];
//    printf("det on CPU:%f\n", res);
//    return 0;
//}
//
////nvcc - arch = sm_35 - rdc = true - o test test2.cu - lcublas_device - lcudadevrt
////. / test
////a is
////1.000000, 0.000000, 1.000000,
////2.000000, 4.000000, 0.000000,
////3.000000, 5.000000, 0.000000,
////det on GPU : 0.000000
////det on CPU : -2.000000