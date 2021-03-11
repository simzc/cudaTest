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

	
	float src_h[n * n * N];			//cpu数组src_h；
	init_array(src_h, N, n);		//cpu数组src_h初始化；

	float* src_d = NULL;			
	checkCudaErrors(cudaMalloc((void**)&src_d, n * n * N * sizeof(float)));						//gpu数组src_d分配内存；
	checkCudaErrors(cudaMemcpy(src_d,src_h, n * n * N * sizeof(float),cudaMemcpyHostToDevice));	//cpu数组src_h复制到gpu端src_d；
	float* A[N];						//cpu上声明存放gpu数组src_d指针的数组；
	set_array_device(A, src_d, N, n);	//数组A寻址；

	float** A_d = NULL;				
	checkCudaErrors(cudaMalloc((void**)&A_d, sizeof(A)));						//gpu上声明存放gpu数组src_d指针的数组；
	checkCudaErrors(cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice));		//cpu端复制A至gpu端；


	float dst_h[n * n * N];			//cpu数组dst_h；
	init_array(dst_h, N, n);		//cpu数组dst_h初始化；

	float* C[N];					//cpu上声明存放gpu数组dst_d指针的数组；
	float* dst_d = NULL;			
	checkCudaErrors(cudaMalloc((void**)&dst_d, n * n * N * sizeof(float)));	//gpu数组dst_d分配内存；
	set_array_device(C, dst_d, N, n);										//数组C寻址；
	
	float** C_d = NULL;
	checkCudaErrors(cudaMalloc((void**)&C_d, sizeof(C)));					//gpu上声明存放gpu数组dst_d指针的数组；
	checkCudaErrors(cudaMemcpy(C_d, C, sizeof(C), cudaMemcpyHostToDevice));	////cpu端复制A至gpu端；

	cublasHandle_t handle;			
	cublasCreate_v2(&handle);					//创建句柄；
	int batchSize = N;							//矩阵块个数；
	int* P = NULL;								//用于记录LU分解的信息；
	int* INFO = NULL;							//用于记录LU分解是否成功；
	checkCudaErrors(cudaMalloc((void**)&P, n * batchSize * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&INFO, batchSize * sizeof(int)));

	checkCudaErrors(cublasSgetrfBatched(handle, n, A_d, n, P, INFO, batchSize));	//LU分解;参数lda为主导维cuBLAS为列主导，即决定了行数；

	int INFOh = 0;
	checkCudaErrors(cudaMemcpy(&INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost));

	if (INFOh == n)//检查矩阵是否奇异；
	{
		fprintf(stderr, "Factorization Failed: Matrix is singular\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cublasSgetriBatched(handle, n, A_d, n, P, C_d, n, INFO, batchSize));//将LU分解的结果转换为逆矩阵；其中C_d保存的是dst_d指针的数组，即二级指针；

	checkCudaErrors(cudaMemcpy(&INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost));
	if (INFOh != 0)//检查矩阵是否奇异；
	{
		fprintf(stderr, "Inversion Failed: Matrix is singular\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaMemcpy(dst_h, dst_d, n*n*N*sizeof(float), cudaMemcpyDeviceToHost));

	//释放内存及销毁句柄；
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