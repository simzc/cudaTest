//#include<iostream>
//#include<cuda_runtime.h>
//#include<helper_cuda.h>
//#include<stdlib.h>
//#include<thrust/extrema.h>
//#include<thrust/sort.h>
//#include<thrust/device_ptr.h>
//#include<thrust/device_vector.h>
//
//int main()
//{
//	int size(10);
//	int* a_host = new int[size];
//
//	for (int i = 0; i < size; i++)
//	{
//		a_host[i] = rand();
//		std::cout << "a_host[" << i << "] = " << a_host[i] << std::endl;
//	}
//
//	int* a_device = NULL;
//	int* max_element_d(NULL);
//	int* max_ele = NULL;
//
//	checkCudaErrors(cudaMalloc((void**)&a_device, size * sizeof(int)));
//	checkCudaErrors(cudaMalloc((void**)&max_element_d, sizeof(int)));
//	checkCudaErrors(cudaMemcpy(a_device, a_host, size * sizeof(int), cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemset(max_element_d, 0, sizeof(int)));
//
//	thrust::sort(
//		thrust::device_ptr<int>(a_device),
//		thrust::device_ptr<int>(a_device + size),
//		thrust::greater<int>());
//
//	max_ele = thrust::max_element(a_host, a_host + size);
//	cudaDeviceSynchronize();
//
//	checkCudaErrors(cudaMemcpy(a_host, a_device, size * sizeof(int), cudaMemcpyDeviceToHost));
//	//checkCudaErrors(cudaMemcpy(max_ele, max_element_d, sizeof(int), cudaMemcpyDeviceToHost));
//
//	std::cout << std::endl;
//
//	for (int i = 0; i < size; i++)
//	{
//		std::cout << "a_device[" << i << "] = " << a_host[i] << std::endl;
//	}
//
//	std::cout << std::endl;
//	std::cout << *max_ele << std::endl;
//	return 0;
//}