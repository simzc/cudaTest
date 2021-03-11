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

	
	float src_h[n * n * N];			//cpu����src_h��
	init_array(src_h, N, n);		//cpu����src_h��ʼ����

	float* src_d = NULL;			
	checkCudaErrors(cudaMalloc((void**)&src_d, n * n * N * sizeof(float)));						//gpu����src_d�����ڴ棻
	checkCudaErrors(cudaMemcpy(src_d,src_h, n * n * N * sizeof(float),cudaMemcpyHostToDevice));	//cpu����src_h���Ƶ�gpu��src_d��
	float* A[N];						//cpu���������gpu����src_dָ������飻
	set_array_device(A, src_d, N, n);	//����AѰַ��

	float** A_d = NULL;				
	checkCudaErrors(cudaMalloc((void**)&A_d, sizeof(A)));						//gpu���������gpu����src_dָ������飻
	checkCudaErrors(cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice));		//cpu�˸���A��gpu�ˣ�


	float dst_h[n * n * N];			//cpu����dst_h��
	init_array(dst_h, N, n);		//cpu����dst_h��ʼ����

	float* C[N];					//cpu���������gpu����dst_dָ������飻
	float* dst_d = NULL;			
	checkCudaErrors(cudaMalloc((void**)&dst_d, n * n * N * sizeof(float)));	//gpu����dst_d�����ڴ棻
	set_array_device(C, dst_d, N, n);										//����CѰַ��
	
	float** C_d = NULL;
	checkCudaErrors(cudaMalloc((void**)&C_d, sizeof(C)));					//gpu���������gpu����dst_dָ������飻
	checkCudaErrors(cudaMemcpy(C_d, C, sizeof(C), cudaMemcpyHostToDevice));	////cpu�˸���A��gpu�ˣ�

	cublasHandle_t handle;			
	cublasCreate_v2(&handle);					//���������
	int batchSize = N;							//����������
	int* P = NULL;								//���ڼ�¼LU�ֽ����Ϣ��
	int* INFO = NULL;							//���ڼ�¼LU�ֽ��Ƿ�ɹ���
	checkCudaErrors(cudaMalloc((void**)&P, n * batchSize * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&INFO, batchSize * sizeof(int)));

	checkCudaErrors(cublasSgetrfBatched(handle, n, A_d, n, P, INFO, batchSize));	//LU�ֽ�;����ldaΪ����άcuBLASΪ����������������������

	int INFOh = 0;
	checkCudaErrors(cudaMemcpy(&INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost));

	if (INFOh == n)//�������Ƿ����죻
	{
		fprintf(stderr, "Factorization Failed: Matrix is singular\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cublasSgetriBatched(handle, n, A_d, n, P, C_d, n, INFO, batchSize));//��LU�ֽ�Ľ��ת��Ϊ���������C_d�������dst_dָ������飬������ָ�룻

	checkCudaErrors(cudaMemcpy(&INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost));
	if (INFOh != 0)//�������Ƿ����죻
	{
		fprintf(stderr, "Inversion Failed: Matrix is singular\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaMemcpy(dst_h, dst_d, n*n*N*sizeof(float), cudaMemcpyDeviceToHost));

	//�ͷ��ڴ漰���پ����
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