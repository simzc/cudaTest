#include<iostream>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<helper_cuda.h>
#include<device_launch_parameters.h>
#include<vector>


//define struct
struct matrix
{
	/*float xx;
	float yx;
	float zx;
	float xy;
	float yy;
	float zy;
	float xz;
	float yz;
	float zz;*/
	
	float p[9];
};

void init_matix(matrix* a,int n)
{
	for (int i = 0; i < n; i++)
	{
		/*a[i].xx = i + 1;
		a[i].xy = i + 2;
		a[i].xz = i + 3;
		a[i].yx = i + 4;
		a[i].yy = i + 5;
		a[i].yz = i + 6;
		a[i].zx = i + 7;
		a[i].zy = i + 8;
		a[i].zz = i + 9;*/
		a->p[0] = 1;
	}
}

__global__ void LU_decompose_cuk(
	matrix* dev_ptr,
	matrix* res_ptr,
	int Num)
{
	cublasHandle_t handle;
	cublasCreate(&handle);



	cublasDestroy(handle);
}

int main()
{
	int N = 6;
	matrix* hos_matrix = new matrix[N];
	matrix* dev_matrix = NULL;
	matrix* dev_result = NULL;

	float p[9];

	matrix a1;
	float* A[] = { a1.p };

	std::vector<float>a;
	
	
	init_matix(hos_matrix, N);

	checkCudaErrors(cudaMalloc((void**)&dev_matrix, N * sizeof(matrix)));
	checkCudaErrors(cudaMalloc((void**)&dev_result, N * sizeof(matrix)));
	checkCudaErrors(cudaMemcpy(dev_matrix, hos_matrix, N * sizeof(matrix), cudaMemcpyHostToDevice));
	cudaMemset()

	dim3 block(128, 1);

	dim3 cuda_grid_size = dim3((N + block.x - 1) / block.x, 1);

	LU_decompose_cuk<<<cuda_grid_size,block>>>
		()
}