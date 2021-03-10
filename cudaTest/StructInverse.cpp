#include<iostream>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<helper_cuda.h>
#include<device_launch_parameters.h>
#include<iomanip>


void init_array(float* a, int N, int n)
{
	for (int i = 0; i < n * n * N; i++) 
	{
		a[i] = i*i + 2.5f;
	}
}

void set_array(float** a, float* b, int N, int n)
{
	for (int i = 0; i < N; i++)
	{
		a[i] =  &b[n * n * i] ;
	}
}

void set_array_device(float** a, float* b, int N, int n)
{
	for (int i = 0; i < N; i++)
	{
		a[i] = &b[n * n * i];
	}
}



int main()
{
	const int N(3), n(3);

	float src_h[n * n * N];
	
	init_array(src_h, N, n);

	float* src_d = NULL;
	checkCudaErrors(cudaMalloc((void**)&src_d, n * n * N * sizeof(float)));
	checkCudaErrors(cudaMemcpy(src_d,src_h, n * n * N * sizeof(float),cudaMemcpyHostToDevice));
	float* A[N];
	set_array_device(A, src_d, N, n);

	float** A_d = NULL;
	//checkCudaErrors(cudaMalloc<float*>(&A_d, sizeof(A)));
	checkCudaErrors(cudaMalloc((void**)&A_d, sizeof(A)));
	checkCudaErrors(cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice));

	//checkCudaErrors(cudaMemcpy(A, A_d, sizeof(A), cudaMemcpyDeviceToHost));

	float dst_h[n * n * N];
	init_array(dst_h, N, n);

	float* C[N];
	float* dst_d = NULL;
	checkCudaErrors(cudaMalloc((void**)&dst_d, n * n * N * sizeof(float)));
	set_array_device(C, dst_d, N, n);
	
	float** C_d = NULL;
	checkCudaErrors(cudaMalloc((void**)&C_d, sizeof(C)));
	//checkCudaErrors(cudaMalloc<float*>(&C_d, sizeof(C)));
	checkCudaErrors(cudaMemcpy(C_d, C, sizeof(C), cudaMemcpyHostToDevice));

	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	int batchSize = N;
	int* P, * INFO;
	checkCudaErrors(cudaMalloc((void**)&P, n * batchSize * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&INFO, batchSize * sizeof(int)));
	checkCudaErrors(cublasSgetrfBatched(handle, n, A_d, n, P, INFO, batchSize));

	int INFOh = 0;
	checkCudaErrors(cudaMemcpy(&INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost));

	if (INFOh == n)
	{
		fprintf(stderr, "Factorization Failed: Matrix is singular\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cublasSgetriBatched(handle, n, A_d, n, P, C_d, n, INFO, batchSize));

	checkCudaErrors(cudaMemcpy(dst_h, dst_d, n*n*N*sizeof(float), cudaMemcpyDeviceToHost));
	//cudaMemcpy(dst_h, dst_d, n * n * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(C_d);
	cudaFree(A_d);
	cudaFree(P), cudaFree(INFO), cublasDestroy_v2(handle);


	int col(1);
	for (int i = 0; i < N; i++)
	{
		std::cout << "The input matrix :" << std::endl;
		for (int j = 0; j < n * n; j++)
		{
			std::cout << std::setiosflags(std::ios::right) <<std::setiosflags(std::ios::fixed) << std::setprecision(4) << src_h[i * n * n + j] << "     ";
			if (col % 3 == 0)
			{
				std::cout << std::endl;
			}
			col++;
		}
		col = 1;
		std::cout << "The output matrix :" << std::endl;
		for (int j = 0; j < n * n; j++)
		{
			std::cout << std::setiosflags(std::ios::right) << std::setiosflags(std::ios::fixed) <<std::setprecision(4)<< dst_h[i * n * n + j] << "     ";
			if (col % 3 == 0)
			{
				std::cout << std::endl;
			}
			col++;
		}
		col = 1;
	}


	return 0;
}