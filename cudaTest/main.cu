////#include<helper_cuda.h>
////#include<cuda_runtime.h>
////#include<device_launch_parameters.h>
////#include<iostream>
////#include<cmath>
////#include<ctime>
////
////
////__global__ void add_cuk(float* x, float* y, float* z, int Num)
////{
////	int index = blockIdx.x * blockDim.x + threadIdx.x;
////
////	if (index < Num)
////	{
////		z[index] = x[index] + y[index];
////	}
////}
////
////void add(float* x, float* y, float* z, int Num)
////{
////	for (int i = 0; i < Num; i++)
////	{
////		z[i] = x[i] + y[i];
////	}
////}
////
////int main()
////{
////	clock_t beg, end;
////
////	
////	int Num = 500000000;
////	int bytes = Num * sizeof(float);
////
////	dim3 block(128, 1);
////	dim3 cuda_grid_size = dim3(static_cast<int>(ceil((Num + block.x - 1) / block.x)), 1);
////
////	float* dx = NULL;
////	float* dy = NULL;
////	float* dz = NULL;
////	float* hx = NULL;
////	float* hy = NULL;
////	float* hz = NULL;
////
////	hx = new float[Num];
////	hy = new float[Num];
////	hz = new float[Num];
////
////	for (int i = 0; i < Num; i++)
////	{
////		hx[i] = 1.01;
////		hy[i] = 1.02;
////		hz[i] = 1.03;
////	}
////
////	cudaMalloc((void**)&dx, bytes);
////	cudaMalloc((void**)&dy, bytes);
////	cudaMalloc((void**)&dz, bytes);
////
////	cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);
////	cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice);
////	cudaMemcpy(dz, hz, bytes, cudaMemcpyHostToDevice);
////
////	beg = clock();
////	add_cuk << <cuda_grid_size, block >> > (dx, dy, dz, Num);
////
////	cudaThreadSynchronize();
////	end = clock();
////
////	cudaFree(dx);
////	cudaFree(dy);
////	cudaFree(dz);
////
////	delete[]hx;
////	delete[]hy;
////	delete[]hz;
////
////	double time = static_cast<double>((end - beg)) / CLOCKS_PER_SEC;
////
////	std::cout << "GPU cost : " << time << std::endl;
////
////
////	clock_t start, finish;
////
////	
////
////	hx = new float[Num];
////	hy = new float[Num];
////	hz = new float[Num];
////
////	for (int i = 0; i < Num; i++)
////	{
////		hx[i] = 1.01;
////		hy[i] = 1.02;
////		hz[i] = 1.03;
////	}
////
////	start = clock();
////	add(hx, hy, hz, Num);
////	
////	finish = clock();
////	delete[]hx;
////	delete[]hy;
////	delete[]hz;
////	
////
////	double time_cpu = static_cast<double>((finish - start)) / CLOCKS_PER_SEC;
////
////	std::cout << "CPU cost : " << time_cpu << std::endl;
////
////	std::cout << "Accelaration rate : " << time_cpu / time << std::endl;
////
////	return 0;
////}
//
//
//
//
//#include<iostream>
//#include<helper_cuda.h>
//#include<cuda_runtime.h>
//#include<device_launch_parameters.h>
//
//__global__ void add_cuk(float* a, int N)
//{
//
//	int index = blockDim.x * blockIdx.x + threadIdx.x;
//	if (index < N)
//	{
//		a[index] += 1;
//	}
//}
//
//
//int main()
//{
//	int n = 30;
//	int bytes = n * sizeof(float);
//
//	float* da = NULL;
//	float* ha = NULL;
//
//	//ha = (float*)malloc(bytes);
//	ha = new float[n];
//	cudaMalloc((void**)&da, bytes);
//
//	for (int i = 0; i < n; i++)
//	{
//		ha[i] = i + 1;
//		std::cout << "ha[ " << i << " ] = " << ha[i] << std::endl;
//	}
//
//	dim3 block(16, 1);
//	dim3 cuda_grid_size((n + block.x - 1) / block.x, 1);
//	cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
//
//	add_cuk << <cuda_grid_size, block >> > (da, n);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(ha, da, bytes, cudaMemcpyDeviceToHost);
//
//	std::cout << "*****************The data from GPUs*****************" << std::endl;
//	for (int i = 0; i < n; i++)
//	{
//		ha[i] = i + 1;
//		std::cout << "ha[ " << i << " ] = " << ha[i] << std::endl;
//	}
//
//	//free(ha);
//	delete[]ha;
//	cudaFree(da);
//	return 0;
//}



//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/generate.h>
//#include <thrust/sort.h>
//#include <thrust/copy.h>
//#include <algorithm>
//#include <vector>
//#include <time.h>
//
//int main(void)
//{
//    thrust::host_vector<int> h_vec(1024 * 1024);
//    std::generate(h_vec.begin(), h_vec.end(), rand);
//
//    std::vector<int> vec(h_vec.size());
//    thrust::copy(h_vec.begin(), h_vec.end(), vec.begin());
//
//    thrust::device_vector<int> d_vec = h_vec;
//
//    clock_t time1, time2;
//
//    time1 = clock();
//    thrust::sort(d_vec.begin(), d_vec.end());
//    time2 = clock();
//    std::cout << (double)(time2 - time1) / CLOCKS_PER_SEC << std::endl;
//
//    time1 = clock();
//    std::sort(vec.begin(), vec.end());
//    time2 = clock();
//    std::cout << (double)(time2 - time1) / CLOCKS_PER_SEC << std::endl;
//
//    time1 = clock();
//    thrust::sort(h_vec.begin(), h_vec.end());
//    time2 = clock();
//    std::cout << (double)(time2 - time1) / CLOCKS_PER_SEC << std::endl;
//
//    return 0;
//}



//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//
//#include <iostream>
//
//int main(void)
//{
//    // H has storage for 4 integers
//    thrust::host_vector<int> H(4);
//
//    // initialize individual elements
//    H[0] = 14;
//    H[1] = 20;
//    H[2] = 38;
//    H[3] = 46;
//
//    // H.size() returns the size of vector H
//    std::cout << "H has size " << H.size() << std::endl;
//
//    // print contents of H
//    for (int i = 0; i < H.size(); i++)
//        std::cout << "H[" << i << "] = " << H[i] << std::endl;
//
//    // resize H
//    H.resize(2);
//
//    std::cout << "H now has size " << H.size() << std::endl;
//
//    // Copy host_vector H to device_vector D
//    thrust::device_vector<int> D = H;
//
//    // elements of D can be modified
//    D[0] = 99;
//    D[1] = 88;
//
//    // print contents of D
//    for (int i = 0; i < D.size(); i++)
//        std::cout << "D[" << i << "] = " << D[i] << std::endl;
//
//    // H and D are automatically deleted when the function returns
//    return 0;
//}



//#include <iostream>
//#include <cstdlib>
//#include <time.h>
//#include <algorithm>
//#include<cuda_runtime.h>
//#include<helper_cuda.h>
//#include<device_launch_parameters.h>
//#include<device_functions.h>
//
//
//using namespace std;
//
//#define NUM_ELEMENT 4096
//#define NUM_LISTS   128
//
//typedef enum {
//    cpu_sort = 0,
//    gpu_merge_1,    //桶排序+单个线程合并
//    gpu_merge_all,  //桶排序+多个线程合并
//    gpu_merge_reduction,    //桶排序+规约合并
//    gpu_merge_reduction_modified,   //桶排序+优化的分块规约合并
//}gpu_calc_type;
//
//template<class T> void c_swap(T& x, T& y) { T tmp = x; x = y; y = tmp; }
//
//unsigned int srcData[NUM_ELEMENT];
//
//void gen_and_shuffle(unsigned int* const srcData)
//{
//    for (int i = 0; i < NUM_ELEMENT; i++)    //生成不重复的数据
//        srcData[i] = i;
//    for (int i = 0; i < NUM_ELEMENT; i++)
//        c_swap(srcData[rand() % NUM_ELEMENT], srcData[i]);
//    return;
//}
//
//void print_data(unsigned int* const srcData)
//{
//    for (int i = 0; i < NUM_ELEMENT; i++)
//    {
//        printf("%4u", srcData[i]);
//        if ((i + 1) % 32 == 0)
//            printf("\n");
//    }
//}
//
//__device__ void copy_data_in_gpu(const unsigned int* const srcData,
//    unsigned int* const dstData,
//    const unsigned int tid)
//{
//    for (int i = 0; i < NUM_ELEMENT; i += NUM_LISTS)
//        dstData[i + tid] = srcData[i + tid];    //行拷贝
//    __syncthreads();
//}
//
//__device__ void radix_sort2(unsigned int* const sort_tmp,
//    unsigned int* const sort_tmp_1,
//    const unsigned int tid) //桶排序
//{
//    for (unsigned int bit_mask = 1; bit_mask > 0; bit_mask <<= 1)    //32位
//    {
//        unsigned int base_cnt_0 = 0;
//        unsigned int base_cnt_1 = 0;
//
//        for (unsigned int i = 0; i < NUM_ELEMENT; i += NUM_LISTS)
//        {
//            if (sort_tmp[i + tid] & bit_mask)  //该位是1，放到sort_tmp_1中
//            {
//                sort_tmp_1[base_cnt_1 + tid] = sort_tmp[i + tid];
//                base_cnt_1 += NUM_LISTS;
//            }
//            else    //该位是0，放到sort_tmp的前面的
//            {
//                sort_tmp[base_cnt_0 + tid] = sort_tmp[i + tid];
//                base_cnt_0 += NUM_LISTS;
//            }
//        }
//
//        for (unsigned int i = 0; i < base_cnt_1; i += NUM_LISTS)  //将sort_tmp_1的数据放到sort_tmp后面
//        {
//            sort_tmp[base_cnt_0 + i + tid] = sort_tmp_1[i + tid];
//        }
//        __syncthreads();
//    }
//}
//
//__device__ void merge_1(unsigned int* const srcData,
//    unsigned int* const dstData,
//    const unsigned int tid) //单线程合并
//{
//    __shared__ unsigned int list_index[NUM_LISTS];  //不使用__shared__的话，就会创建在寄存器中，寄存器空间不够则会创建在全局内存中，很慢
//    list_index[tid] = tid;    //使用多线程初始化
//    __syncthreads();
//
//    if (tid == 0)    //使用单个线程merge
//    {
//        for (int i = 0; i < NUM_ELEMENT; i++)    //执行NUM_ELEMENT次
//        {
//            unsigned int min_val = 0xFFFFFFFF;
//            unsigned int min_idx = 0;
//            for (int j = 0; j < NUM_LISTS; j++)  //遍历每个list的头指针
//            {
//                if (list_index[j] >= NUM_ELEMENT)    //列表已经走完则跳过
//                    continue;
//                if (srcData[list_index[j]] < min_val)
//                {
//                    min_val = srcData[list_index[j]];
//                    min_idx = j;
//                }
//            }
//            list_index[min_idx] += NUM_LISTS;   //最小的那个指针向后一位
//            dstData[i] = min_val;
//        }
//    }
//}
//
//__device__ void merge_atomicMin(unsigned int* const srcData,
//    unsigned int* const dstData,
//    const unsigned int tid)   //多线程合并
//{
//    unsigned int self_index = tid;
//
//    for (int i = 0; i < NUM_ELEMENT; i++)
//    {
//        __shared__ unsigned int min_val;
//        unsigned int self_data = 0xFFFFFFFF;
//
//        if (self_index < NUM_ELEMENT)
//        {
//            self_data = srcData[self_index];
//        }
//
//        __syncthreads();
//
//        atomicMin(&min_val, self_data);
//
//        if (min_val == self_data)
//        {
//            self_index += NUM_LISTS;
//            dstData[i] = min_val;
//            min_val = 0xFFFFFFFF;
//        }
//
//    }
//}
//
//__device__ void merge_two(unsigned int* const srcData,
//    unsigned int* dstData,
//    const unsigned int tid) //归约合并
//{
//    unsigned int self_index = tid;
//    __shared__ unsigned int data[NUM_LISTS];
//    __shared__ unsigned int tid_max;
//
//    for (int i = 0; i < NUM_ELEMENT; i++)
//    {
//        data[tid] = 0xFFFFFFFF;
//
//        if (self_index < NUM_ELEMENT)
//        {
//            data[tid] = srcData[self_index];
//        }
//
//        if (tid == 0)
//        {
//            tid_max = NUM_LISTS >> 1;
//        }
//
//        __syncthreads();
//
//        while (tid_max > 0)
//        {
//            if (tid < tid_max)
//            {
//                if (data[tid] > data[tid + tid_max])   //小的换到前半段
//                {
//                    data[tid] = data[tid + tid_max];
//                }
//            }
//            __syncthreads();
//            if (tid == 0)    //不清楚书里面为什么不让单一线程处理共享变量，不会出问题吗？
//            {
//                tid_max >>= 1;
//            }
//            __syncthreads();
//        }
//
//        if (srcData[self_index] == data[0])
//        {
//            dstData[i] = data[0];
//            self_index += NUM_LISTS;
//        }
//    }
//}
//
//#define REDUCTION_SIZE  8
//#define REDUCTION_SHIFT 3
//
//__device__ void merge_final(unsigned int* const srcData,
//    unsigned int* const dstData,
//    const unsigned int tid) //分块的归约合并
//{
//    __shared__ unsigned int min_val_reduction[NUM_LISTS / REDUCTION_SIZE];
//    unsigned int s_tid = tid >> REDUCTION_SHIFT;
//    unsigned int self_index = tid;
//    __shared__ unsigned int min_val;
//
//    for (int i = 0; i < NUM_ELEMENT; i++)
//    {
//        unsigned int self_data = 0xFFFFFFFF;
//
//        if (self_index < NUM_ELEMENT)
//        {
//            self_data = srcData[self_index];
//        }
//
//        if (tid < NUM_LISTS / REDUCTION_SIZE)
//        {
//            min_val_reduction[tid] = 0xFFFFFFFF;
//        }
//
//        __syncthreads();
//
//        atomicMin(&(min_val_reduction[s_tid]), self_data);  //分块归约
//
//        __syncthreads();
//
//        if (tid == 0)
//        {
//            min_val = 0xFFFFFFFF;
//        }
//
//        __syncthreads();
//
//        if (tid < NUM_LISTS / REDUCTION_SIZE)
//        {
//            atomicMin(&min_val, min_val_reduction[tid]);    //归约起来的值再归约
//        }
//
//        __syncthreads();
//
//        if (min_val == self_data)
//        {
//            dstData[i] = min_val;
//            self_index += NUM_LISTS;
//            min_val = 0xFFFFFFFF;
//        }
//
//    }
//}
//
//__global__ void sortincuda(unsigned int* const data, gpu_calc_type type)
//{
//    const unsigned int tid = threadIdx.x;
//    __shared__ unsigned int sort_tmp[NUM_ELEMENT], sort_tmp_1[NUM_ELEMENT];
//
//    copy_data_in_gpu(data, sort_tmp, tid);  //因为数据要经常被读取写入，因此拷贝数据到共享内存中以加速
//
//    radix_sort2(sort_tmp, sort_tmp_1, tid); //桶排序
//
//    switch (type)
//    {
//    case cpu_sort:  break;
//    case gpu_merge_1: merge_1(sort_tmp, data, tid); break;  //单线程合并
//    case gpu_merge_all: merge_atomicMin(sort_tmp, data, tid); break;    //多线程合并
//    case gpu_merge_reduction: merge_two(sort_tmp, data, tid); break;    //两两归约合并
//    case gpu_merge_reduction_modified: merge_final(sort_tmp, data, tid); break; //分块归约合并
//    default: break;
//    }
//}
//
//int main(void)
//{
//    gen_and_shuffle(srcData);
//    //print_data(srcData);
//
//    //printf("\n\n");
//
//    unsigned int* gpu_srcData;
//
//    cudaMalloc((void**)&gpu_srcData, sizeof(unsigned int) * NUM_ELEMENT);
//    cudaMemcpy(gpu_srcData, srcData, sizeof(unsigned int) * NUM_ELEMENT, cudaMemcpyHostToDevice);
//
//    clock_t start, end;
//
//    for (gpu_calc_type type = cpu_sort; type <= gpu_merge_reduction_modified; type = (gpu_calc_type)(type + 1))
//    {
//        if (type != cpu_sort)    //gpu排序
//        {
//            start = clock();
//            sortincuda << <1, NUM_LISTS >> > (gpu_srcData, type);
//            cudaDeviceSynchronize();
//            end = clock();
//            printf("type %d use time %.8lf\n", type, (double)(end - start) / CLOCKS_PER_SEC);
//        }
//        else    //cpu排序
//        {
//            start = clock();
//            sort(srcData, srcData + NUM_ELEMENT);
//            end = clock();
//            printf("type %d use time %.8lf\n", type, (double)(end - start) / CLOCKS_PER_SEC);
//        }
//    }
//
//    cudaMemcpy(srcData, gpu_srcData, sizeof(unsigned int) * NUM_ELEMENT, cudaMemcpyDeviceToHost);
//    //print_data(srcData);
//
//    cudaFree(gpu_srcData);
//
//
//    return 0;
//
//}



//#include<cuda.h>
//#include<cuda_runtime.h>
//#include<device_launch_parameters.h>
//#include<helper_cuda.h>
//#include<thrust/device_vector.h>
//#include<thrust/host_vector.h>
//#include<thrust/sort.h>
//#include<iostream>
//
//
//__global__ void copy_cuk(float* d_a, int Num)
//{
//	int index = blockDim.x * blockIdx.x + threadIdx.x;
//
//	if (index < Num)
//	{
//		d_a[index] = index;
//	}
//}
//
//int main()
//{
//	int Num = 10;
//
//	float* d_a = NULL;
//	float* h_a = NULL;
//
//	h_a = new float[Num];
//	cudaMalloc((void**)&d_a, Num * sizeof(float));
//
//	//thrust::host_vector<int> Host_array(Num);
//
//	//for (int i = 0; i < Num; i++)
//	//{
//	//	Host_array[i] = Num - i;
//	//}
//
//	//thrust::device_vector<int> Device_array = Host_array;
//	thrust::device_vector<int> Device_array(Num);
//
//
//
//	dim3 block(128, 1);
//	dim3 grid_size((Num + block.x - 1) / block.x, 1);
//
//	copy_cuk << <grid_size, block >> > ( d_a, Num);
//
//	cudaDeviceSynchronize();
//	cudaMemcpy(h_a, d_a, Num * sizeof(float), cudaMemcpyDeviceToHost);
//
//	for (int i = 0; i < Num; i++)
//	{
//		//d_a[i] = Num - i;
//		Device_array[i] = h_a[i];
//	}
//
//	thrust::sort(Device_array.begin(), Device_array.end(), thrust::greater<int>());
//
//	for (int i = 0; i < Device_array.size(); i++)
//	{
//		std::cout << "D[" << i << "] = " << Device_array[i] << std::endl;
//	}
//
//	delete[]h_a;
//	cudaFree(d_a);
//	Device_array.clear();
//
//	return 0;
//}



//#include<iostream>
//#include<cuda_runtime.h>
//#include<device_launch_parameters.h>
//#include<helper_cuda.h>
//#include<thrust/device_vector.h>
//#include<thrust/host_vector.h>
//#include<thrust/sort.h>
//
//int main()
//{
//	float* h_a = NULL;
//	float* d_a = NULL;
//
//	int n = 10;
//
//	h_a = (float*)malloc(sizeof(float) * n);
//	cudaMalloc((void**)&d_a, sizeof(float) * n);
//
//	for (int i = 0; i < n; i++)
//	{
//		h_a[i] = i + 1;
//	}
//
//	cudaMemcpy(d_a, h_a, sizeof(float) * n, cudaMemcpyHostToDevice);
//
//	float* hh_a = new float[n];
//	cudaMemcpy(hh_a, d_a, sizeof(float) * n, cudaMemcpyDeviceToHost);
//
//
//	for (int i = 0; i < n; i++)
//	{
//		std::cout << "hh_a[" << i << "] = " << hh_a[i] << std::endl;
//	}
//
//	delete[]hh_a;
//	cudaFree(d_a);
//	free(h_a);
//	
//	return 0;
//}



////主机内存拷贝与for循环赋值速度对比；
//#include<iostream>
//#include<ctime>
//
//int main()
//{
//	int num = 100000000;
//
//	float* a = new float[num];
//	float* b = new float[num];
//
//	for (int i = 0; i < num; i++)
//	{
//		a[i] = 100.0f;
//	}
//
//	std::clock_t start, finish;
//
//	start = std::clock();
//	for (int i = 0; i < num; i++)
//	{
//		b[i] = a[i];
//	}
//	finish = std::clock();
//
//	std::cout << "For loop cost : " << double(finish - start) / (CLOCKS_PER_SEC) << std::endl;
//
//	std::clock_t beg, end;
//
//	beg = std::clock();
//	memcpy(b, a, num * sizeof(int));
//	end = std::clock();
//
//	std::cout << "Copy cost : " << double(end - beg) / (CLOCKS_PER_SEC) << std::endl;
//
//	delete[]a;
//	delete[]b;
//
//	return 0;
//}

//
//#include<iostream>
//#include<string>
//#include<vector>
//#include<thrust/sort.h>
//#include<thrust/device_ptr.h>
//#include<cuda_runtime.h>
//#include<helper_cuda.h>
//#include<cooperative_groups.h>
//
//
//int main()
//{
//	int* h_a = NULL;
//	int* d_a = NULL;
//	int* h_idx = NULL;
//	int* d_idx = NULL;
//
//	h_a = new int[10];
//	h_idx = new int[10];
//	cudaMalloc((void**)&d_a, 10 * sizeof(int));
//	cudaMalloc((void**)&d_idx, 10 * sizeof(int));
//
//	//Initialize h_a[]
//	for (int i = 0; i < 10; i++)
//	{
//		h_a[i] = rand();
//		h_idx[i] = i + 1;
//
//		std::cout << "h_idx[" << i << "] = " << h_idx[i] << "\t";
//		std::cout << "h_a[" << i << "] = " << h_a[i] << std::endl;
//	}
//
//	std::cout << std::endl;
//	std::cout << "----------------------------------------------" << std::endl;
//	std::cout << "------------------Sort Array------------------" << std::endl;
//	std::cout << "----------------------------------------------" << std::endl;
//	std::cout << std::endl;
//
//	cudaMemcpy(d_a, h_a, 10 * sizeof(int), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_idx, h_idx, 10 * sizeof(int), cudaMemcpyHostToDevice);
//
//	thrust::sort_by_key(
//		thrust::device_ptr<int>(d_a),
//		thrust::device_ptr<int>(d_a + 10),
//		thrust::device_ptr<int>(d_idx));
//
//	cudaMemcpy(h_a, d_a, 10 * sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_idx, d_idx, 10 * sizeof(int), cudaMemcpyDeviceToHost);
//
//
//	for (int i = 0; i < 10; i++)
//	{
//		std::cout << "h_idx[" << i << "] = " << h_idx[i] << "\t";
//		std::cout << "h_a[" << i << "] = " << h_a[i] << std::endl;
//	}
//
//	//Free memory
//	delete[]h_a;
//	delete[]h_idx;
//
//	cudaFree(d_a);
//	cudaFree(d_idx);
//
//	return 0;
//}




//#include<iostream>
//#include<cuda_runtime.h>
//#include<string>
//
//
//int main() {
//
//	int* h_a = NULL;
//	int* h_b = NULL;
//	int* d_a = NULL;
//	int* d_b = NULL;
//
//	int num = 10;
//	size_t size = num * sizeof(int);
//
//	h_a = new int[10];
//	h_b = new int[10];
//	cudaMalloc((void**)&d_a, size);
//	cudaMalloc((void**)&d_b, size);
//
//	for (int i = 0; i < 10; i++)
//	{
//		h_a[i] = i;
//	}
//
//	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
//	d_b = d_a;
//	cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
//
//	for (int i = 0; i < 10; i++)
//	{
//		std::cout << h_b[i] << std::endl;
//	}
//
//	std::cout << "Hello World!" << std::endl;
//
//	delete[] h_a;
//	delete[] h_b;
//	cudaFree(d_a);
//	cudaFree(d_b);
//
//	return 0;
//}


//#include<iostream>
//#include<thrust/sort.h>
//#include<vector>
//
//int main()
//{
//
//	std::vector<int> a;
//	std::vector<int> index;
//
//	a.resize(8);
//	index.resize(a.size());
//
//	for (int i = 0; i < a.size(); i++)
//	{
//		a[i] = i + 1;
//		index[i] = i + 1;
//
//		std::cout << a[i] << std::endl;
//	}
//
//	thrust::sort_by_key(&a[0], &a[0] + 4, &index[0], thrust::greater<int>());
//
//	std::cout << std::endl;
//	for (int i = 0; i < a.size(); i++)
//	{
//
//		std::cout << index[i] << std::endl;
//	}
//}


///******************************************************************************
// * Copyright (c) 2011, Duane Merrill.  All rights reserved.
// * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
// *
// * Redistribution and use in source and binary forms, with or without
// * modification, are permitted provided that the following conditions are met:
// *     * Redistributions of source code must retain the above copyright
// *       notice, this list of conditions and the following disclaimer.
// *     * Redistributions in binary form must reproduce the above copyright
// *       notice, this list of conditions and the following disclaimer in the
// *       documentation and/or other materials provided with the distribution.
// *     * Neither the name of the NVIDIA CORPORATION nor the
// *       names of its contributors may be used to endorse or promote products
// *       derived from this software without specific prior written permission.
// *
// * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
// * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// *
// ******************************************************************************/
//
// /******************************************************************************
//  * Simple example of DeviceRadixSort::SortPairs().
//  *
//  * Sorts an array of float keys paired with a corresponding array of int values.
//  *
//  * To compile using the command line:
//  *   nvcc -arch=sm_XX example_device_radix_sort.cu -I../.. -lcudart -O3
//  *
//  ******************************************************************************/
//
//  // Ensure printing of CUDA runtime errors to console
//#define CUB_STDERR
//
//#include <stdio.h>
//#include <algorithm>
//
//#include <cub/util_allocator.cuh>
//#include <cub/device/device_radix_sort.cuh>
//
//#include "E:/Program Files/cub-1.8.0/test/test_util.h"
//
//using namespace cub;
//
//
////---------------------------------------------------------------------
//// Globals, constants and typedefs
////---------------------------------------------------------------------
//
//bool                    g_verbose = false;  // Whether to display input/output to console
//CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory
//
//
////---------------------------------------------------------------------
//// Test generation
////---------------------------------------------------------------------
//
///**
// * Simple key-value pairing for floating point types.  Distinguishes
// * between positive and negative zero.
// */
//struct Pair
//{
//    float   key;
//    int     value;
//
//    bool operator<(const Pair& b) const
//    {
//        if (key < b.key)
//            return true;
//
//        if (key > b.key)
//            return false;
//
//        // Return true if key is negative zero and b.key is positive zero
//        unsigned int key_bits = *reinterpret_cast<unsigned*>(const_cast<float*>(&key));
//        unsigned int b_key_bits = *reinterpret_cast<unsigned*>(const_cast<float*>(&b.key));
//        unsigned int HIGH_BIT = 1u << 31;
//
//        return ((key_bits & HIGH_BIT) != 0) && ((b_key_bits & HIGH_BIT) == 0);
//    }
//};
//
//
///**
// * Initialize key-value sorting problem.
// */
//void Initialize(
//    float* h_keys,
//    int* h_values,
//    float* h_reference_keys,
//    int* h_reference_values,
//    int             num_items)
//{
//    Pair* h_pairs = new Pair[num_items];
//
//    for (int i = 0; i < num_items; ++i)
//    {
//        RandomBits(h_keys[i]);
//        RandomBits(h_values[i]);
//        h_pairs[i].key = h_keys[i];
//        h_pairs[i].value = h_values[i];
//    }
//
//    if (g_verbose)
//    {
//        printf("Input keys:\n");
//        DisplayResults(h_keys, num_items);
//        printf("\n\n");
//
//        printf("Input values:\n");
//        DisplayResults(h_values, num_items);
//        printf("\n\n");
//    }
//
//    std::stable_sort(h_pairs, h_pairs + num_items);
//
//    for (int i = 0; i < num_items; ++i)
//    {
//        h_reference_keys[i] = h_pairs[i].key;
//        h_reference_values[i] = h_pairs[i].value;
//    }
//
//    delete[] h_pairs;
//}
//
//
////---------------------------------------------------------------------
//// Main
////---------------------------------------------------------------------
//
///**
// * Main
// */
//int main(int argc, char** argv)
//{
//    int num_items = 150;
//
//    // Initialize command line
//    CommandLineArgs args(argc, argv);
//    g_verbose = args.CheckCmdLineFlag("v");
//    args.GetCmdLineArgument("n", num_items);
//
//    // Print usage
//    if (args.CheckCmdLineFlag("help"))
//    {
//        printf("%s "
//            "[--n=<input items> "
//            "[--device=<device-id>] "
//            "[--v] "
//            "\n", argv[0]);
//        exit(0);
//    }
//
//    // Initialize device
//    CubDebugExit(args.DeviceInit());
//
//    printf("cub::DeviceRadixSort::SortPairs() %d items (%d-byte keys %d-byte values)\n",
//        num_items, int(sizeof(float)), int(sizeof(int)));
//    fflush(stdout);
//
//    // Allocate host arrays
//    float* h_keys = new float[num_items];
//    float* h_reference_keys = new float[num_items];
//    int* h_values = new int[num_items];
//    int* h_reference_values = new int[num_items];
//
//    // Initialize problem and solution on host
//    Initialize(h_keys, h_values, h_reference_keys, h_reference_values, num_items);
//
//    // Allocate device arrays
//    DoubleBuffer<float> d_keys;
//    DoubleBuffer<int>   d_values;
//    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(float) * num_items));
//    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(float) * num_items));
//    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(int) * num_items));
//    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(int) * num_items));
//
//    // Allocate temporary storage
//    size_t  temp_storage_bytes = 0;
//    void* d_temp_storage = NULL;
//
//    CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));
//    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
//
//    // Initialize device arrays
//    CubDebugExit(cudaMemcpy(d_keys.d_buffers[d_keys.selector], h_keys, sizeof(float) * num_items, cudaMemcpyHostToDevice));
//    CubDebugExit(cudaMemcpy(d_values.d_buffers[d_values.selector], h_values, sizeof(int) * num_items, cudaMemcpyHostToDevice));
//
//    // Run
//    CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));
//
//    // Check for correctness (and display results, if specified)
//    int compare = CompareDeviceResults(h_reference_keys, d_keys.Current(), num_items, true, g_verbose);
//    printf("\t Compare keys (selector %d): %s\n", d_keys.selector, compare ? "FAIL" : "PASS");
//    AssertEquals(0, compare);
//    compare = CompareDeviceResults(h_reference_values, d_values.Current(), num_items, true, g_verbose);
//    printf("\t Compare values (selector %d): %s\n", d_values.selector, compare ? "FAIL" : "PASS");
//    AssertEquals(0, compare);
//
//    // Cleanup
//    if (h_keys) delete[] h_keys;
//    if (h_reference_keys) delete[] h_reference_keys;
//    if (h_values) delete[] h_values;
//    if (h_reference_values) delete[] h_reference_values;
//
//    if (d_keys.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[0]));
//    if (d_keys.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
//    if (d_values.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[0]));
//    if (d_values.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[1]));
//    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
//
//    printf("\n\n");
//
//    return 0;
//}
//
//



//// Ensure printing of CUDA runtime errors to console
//#define CUB_STDERR
//#include <stdio.h>
//#include <cub/util_allocator.cuh>
//#include <cub/device/device_scan.cuh>
//#include "E:\Program Files\cub-1.8.0\test/test_util.h"
//using namespace cub;
////---------------------------------------------------------------------
//// Globals, constants and typedefs
////---------------------------------------------------------------------
//bool                    g_verbose = false;  // Whether to display input/output to console
//CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory
////---------------------------------------------------------------------
//// Test generation
////---------------------------------------------------------------------
//void Initialize(
//    int* h_in,
//    int          num_items)
//{
//    for (int i = 0; i < num_items; ++i)
//        h_in[i] = i;
//    if (g_verbose)
//    {
//        printf("Input:\n");
//        DisplayResults(h_in, num_items);
//        printf("\n\n");
//    }
//}
//int Solve(
//    int* h_in,
//    int* h_reference,
//    int             num_items)
//{
//    int inclusive = 0;
//    int aggregate = 0;
//    for (int i = 0; i < num_items; ++i)
//    {
//        h_reference[i] = inclusive;
//        inclusive += h_in[i];
//        aggregate += h_in[i];
//    }
//    return aggregate;
//}
////---------------------------------------------------------------------
//// Main
////---------------------------------------------------------------------
//int main(int argc, char** argv)
//{
//    int num_items = 150;
//    // Initialize command line
//    CommandLineArgs args(argc, argv);
//    g_verbose = args.CheckCmdLineFlag("v");
//    args.GetCmdLineArgument("n", num_items);
//    // Print usage
//    if (args.CheckCmdLineFlag("help"))
//    {
//        printf("%s "
//            "[--n=<input items> "
//            "[--device=<device-id>] "
//            "[--v] "
//            "\n", argv[0]);
//        exit(0);
//    }
//    // Initialize device
//    CubDebugExit(args.DeviceInit());
//    printf("cub::DeviceScan::ExclusiveSum %d items (%d-byte elements)\n",
//        num_items, (int)sizeof(int));
//    fflush(stdout);
//    // Allocate host arrays
//    int* h_in = new int[num_items];
//    int* h_reference = new int[num_items];
//    // Initialize problem and solution
//    Initialize(h_in, num_items);
//    Solve(h_in, h_reference, num_items);
//    // Allocate problem device arrays
//    int* d_in = NULL;
//    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(int) * num_items));
//    // Initialize device input
//    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));
//    // Allocate device output array
//    int* d_out = NULL;
//    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(int) * num_items));
//    // Allocate temporary storage
//    void* d_temp_storage = NULL;
//    size_t          temp_storage_bytes = 0;
//    CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
//    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
//    // Run
//    CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
//    // Check for correctness (and display results, if specified)
//    int compare = CompareDeviceResults(h_reference, d_out, num_items, true, g_verbose);
//    printf("\t%s", compare ? "FAIL" : "PASS");
//    AssertEquals(0, compare);
//    // Cleanup
//    if (h_in) delete[] h_in;
//    if (h_reference) delete[] h_reference;
//    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
//    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
//    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
//    printf("\n\n");
//    return 0;
//}


//#include<iostream>
//#include<cuda_runtime.h>
//
//struct Point
//{
//	float x;
//	float y;
//	float z;
//
//	Point& operator=(const float& v1)
//	{
//		x = v1;
//		y = v1;
//		z = v1;
//		
//		return *this;
//	}
//};
//
//
//int main()
//{
//	int x(1), y(2), z(3);
//
//	Point a = 0.0;
//
//	return 0;
//}



//#include <stdio.h>
//#include <stdlib.h>
//#include <cublas_v2.h>
//#include<cublas.h>
//#include<math.h>
//
//
//#define cudacall(call)                                                                                                          \
//    do                                                                                                                          \
//    {                                                                                                                           \
//        cudaError_t err = (call);                                                                                               \
//        if(cudaSuccess != err)                                                                                                  \
//        {                                                                                                                       \
//            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
//            cudaDeviceReset();                                                                                                  \
//            exit(EXIT_FAILURE);                                                                                                 \
//        }                                                                                                                       \
//    }                                                                                                                           \
//    while (0)
//
//#define cublascall(call)                                                                                        \
//    do                                                                                                          \
//    {                                                                                                           \
//        cublasStatus_t status = (call);                                                                         \
//        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
//        {                                                                                                       \
//            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
//            cudaDeviceReset();                                                                                  \
//            exit(EXIT_FAILURE);                                                                                 \
//        }                                                                                                       \
//                                                                                                                \
//    }                                                                                                           \
//    while(0)
//
//////////////////////////////////////////////////////////////////////////
////
////
//////////////////////////////////////////////////////////////////////////
//void invert(float** src, float** dst, int n, int batchSize)
//{
//	cublasHandle_t handle;
//	cublascall(cublasCreate_v2(&handle));
//
//	int* P, * INFO;
//
//	cudacall(cudaMalloc(&P, n * batchSize * sizeof(int)));
//	cudacall(cudaMalloc(&INFO, batchSize * sizeof(int)));
//
//	int lda = n;
//
//	float** A = (float**)malloc(batchSize * sizeof(float*));
//	float** A_d, * A_dflat;
//
//	cudacall(cudaMalloc(&A_d, batchSize * sizeof(float*)));
//	cudacall(cudaMalloc(&A_dflat, n * n * batchSize * sizeof(float)));
//
//	A[0] = A_dflat;
//	for (int i = 1; i < batchSize; i++)
//		A[i] = A[i - 1] + (n * n);
//
//	cudacall(cudaMemcpy(A_d, A, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
//
//	for (int i = 0; i < batchSize; i++)
//		cudacall(cudaMemcpy(A_dflat + (i * n * n), src[i], n * n * sizeof(float), cudaMemcpyHostToDevice));
//
//
//	cublascall(cublasSgetrfBatched(handle, n, A_d, lda, P, INFO, batchSize));
//
//
//	//int INFOh[batchSize];
//	int INFOh[batchSize];
//	cudacall(cudaMemcpy(INFOh, INFO, batchSize * sizeof(int), cudaMemcpyDeviceToHost));
//
//	for (int i = 0; i < batchSize; i++)
//		if (INFOh[i] != 0)
//		{
//			fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
//			cudaDeviceReset();
//			exit(EXIT_FAILURE);
//		}
//
//	float** C = (float**)malloc(batchSize * sizeof(float*));
//	float** C_d, * C_dflat;
//
//	cudacall(cudaMalloc(&C_d, batchSize * sizeof(float*)));
//	cudacall(cudaMalloc(&C_dflat, n * n * batchSize * sizeof(float)));
//	C[0] = C_dflat;
//	for (int i = 1; i < batchSize; i++)
//		C[i] = C[i - 1] + (n * n);
//	cudacall(cudaMemcpy(C_d, C, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
//	cublascall(cublasSgetriBatched(handle, n, (const float**)A_d, lda, P, C_d, lda, INFO, batchSize));
//
//	cudacall(cudaMemcpy(INFOh, INFO, batchSize * sizeof(int), cudaMemcpyDeviceToHost));
//
//	for (int i = 0; i < batchSize; i++)
//		if (INFOh[i] != 0)
//		{
//			fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
//			cudaDeviceReset();
//			exit(EXIT_FAILURE);
//		}
//	for (int i = 0; i < batchSize; i++)
//		cudacall(cudaMemcpy(dst[i], C_dflat + (i * n * n), n * n * sizeof(float), cudaMemcpyDeviceToHost));
//
//	cudaFree(A_d); cudaFree(A_dflat); free(A);
//	cudaFree(C_d); cudaFree(C_dflat); free(C);
//	cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle);
//}
//
//
//////////////////////////////////////////////////////////////////////////
////
////
//////////////////////////////////////////////////////////////////////////
//
//
//__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
//{
//	//2D Thread ID
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
//
//	//Pvalue stores the Pd element that is computed by the thread
//	float Pvalue = 0;
//	if (col < Width && row < Width)
//	{
//		for (int k = 0; k < Width; ++k)
//		{
//			float Mdelement = Md[row * Width + k];
//			float Ndelement = Nd[k * Width + col];
//			Pvalue += (Mdelement * Ndelement);
//
//		}
//		Pd[row * Width + col] = Pvalue;
//	}
//}
//
//
//
//void mul(float* M, float* N, int Width)
//{
//
//	float* P = (float*)malloc(Width * Width * sizeof(float));
//	float* Md, * Nd, * Pd;
//
//
//
//	unsigned long int size = Width * Width * sizeof(float);
//
//
//	//Transfer M and N to device memory
//	cudaMalloc((void**)&Md, size);
//	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
//
//	cudaMalloc((void**)&Nd, size);
//	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
//
//	//Allocate P on the device
//	cudaMalloc((void**)&Pd, size);
//
//	//Setup the execution configuration
//	dim3 dimBlock(Width, Width);
//	dim3 dimGrid(1, 1);
//
//
//	if (Width * Width > 1024)
//	{
//		//printf("\n\n enter inside if condi\n\n");
//
//		dimGrid.x = (Width - 1) / 32 + 1;
//		dimGrid.y = (Width - 1) / 32 + 1;
//
//		dimBlock.x = 32;
//		dimBlock.y = 32;
//
//
//
//	}
//
//
//	//Launch the device computation threads!
//	MatrixMulKernel << <dimGrid, dimBlock >> > (Md, Nd, Pd, Width);
//
//	//Transfer P from device to host
//	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
//
//	//Free device matrices
//	cudaFree(Md);
//	cudaFree(Nd);
//	cudaFree(Pd);
//
//	int i;
//
//	fprintf(stdout, "\n\n");
//
//	if (Width < 11)
//	{
//
//
//		fprintf(stdout, "\n\nMatrix Multiplication, M x Inv(M) :\n\n");
//		for (i = 0; i < Width * Width; i++)
//		{
//			if (P[i])
//				fprintf(stdout, "%10f ", P[i]);
//			else
//				fprintf(stdout, "%9f ", P[i]);
//
//
//
//
//			if ((i + 1) % Width == 0)
//				fprintf(stdout, "\n");
//		}
//
//
//	}
//	else
//	{
//		FILE* fp;
//
//		fp = fopen("Mat_Inv_out", "a");
//
//		if (!fp)
//		{
//			fprintf(stderr, "Failed to open matAdata.\n");
//			exit(1);
//		}
//		fprintf(fp, "\n\nMatrix Multiplication, M x Inv(M) :\n\n");
//		for (i = 0; i < Width * Width; i++)
//		{
//			if (P[i])
//				fprintf(fp, "%10f ", P[i]);
//			else
//				fprintf(fp, "%9f ", P[i]);
//
//			if ((i + 1) % Width == 0)
//				fprintf(fp, "\n");
//		}
//		fclose(fp);
//	}
//
//
//	//printf("\n Matrix multiplication completed !!\n\n"); 
//	free(M);
//	free(N);
//	free(P);
//
//}
//
//
//////////////////////////////////////////////////////////////////////////
////
////
//////////////////////////////////////////////////////////////////////////
//
//
//void fill(float* h, int w)
//{
//
//	unsigned int i, num;
//	int divide;
//	FILE* f;
//
//	f = fopen("/dev/urandom", "r");
//	if (!f) {
//		fprintf(stderr, "Failed open file\n");
//		exit(1);
//	}
//	for (i = 0; i < w * w; i++)
//	{
//		fread(&num, sizeof(unsigned int), 1, f);
//		fread(&divide, sizeof(int), 1, f);
//		h[i] = ((float)num) / ((float)divide);
//		//scanf("%f",&h[i]);
//	}
//	fclose(f);
//	/*
//		unsigned int i;
//		srand((unsigned int)time(NULL));
//		for(i=0; i< w*w; i++)
//		{
//			h[i] = ((float)rand()/(float)(RAND_MAX)) * 99;
//			//scanf("%f",&h[i]);
//		}
//
//	*/
//
//}
//
//////////////////////////////////////////////////////////////////////////
////
////
//////////////////////////////////////////////////////////////////////////
//
//void test_invert(int n)
//{
//
//	//printf("Enter the order of the square matrix :");
//	//scanf("%d",&n);
//	const int mybatch = 1;
//
//
//	//float* mat1[n * n];
//	float mat1_size = sizeof(float) * n * n;
//	float* mat1 = (float*)malloc(mat1_size);
//
//	fill(mat1, n);
//
//	float* result_flat = (float*)malloc(mybatch * n * n * sizeof(float));
//	float** results = (float**)malloc(mybatch * sizeof(float*));
//
//	for (int i = 0; i < mybatch; i++)
//		results[i] = result_flat + (i * n * n);
//
//	float** inputs = (float**)malloc(mybatch * sizeof(float*));
//
//	//inputs[0]  = zero_pivot;
//
//	inputs[0] = mat1;
//
//
//	invert(inputs, results, n, mybatch);
//
//	if (n < 11)
//	{
//
//		for (int qq = 0; qq < mybatch; qq++)
//		{
//			if (mybatch == 1)
//				fprintf(stdout, "Input Matrix, M :\n\n");
//			else
//				fprintf(stdout, "Input Matrix %d:\n\n", qq);
//
//			for (int i = 0; i < n; i++)
//			{
//				for (int j = 0; j < n; j++)
//				{
//					if (inputs[qq][i * n + j])
//						fprintf(stdout, "%12f ", inputs[qq][i * n + j]);
//					else
//						fprintf(stdout, "%11f ", inputs[qq][i * n + j]);
//				}
//				fprintf(stdout, "\n");
//			}
//		}
//		fprintf(stdout, "\n\n");
//
//
//
//
//		for (int qq = 0; qq < mybatch; qq++)
//		{
//
//			if (mybatch == 1)
//				fprintf(stdout, "Inverse of the Input Matrix, Inv(M):\n\n");
//			else
//				fprintf(stdout, "Inverse Matrix %d:\n\n", qq);
//			for (int i = 0; i < n; i++)
//			{
//				for (int j = 0; j < n; j++)
//				{
//					if (results[qq][i * n + j])
//						fprintf(stdout, "%10f ", results[qq][i * n + j]);
//					else
//						fprintf(stdout, "%9f ", results[qq][i * n + j]);
//
//				}
//				fprintf(stdout, "\n");
//			}
//		}
//	}
//
//
//	else // order of the matrix is more than 10 x 10 then output the results in the file
//	{
//		printf("\nThe order of matrix is too large to display in terminal\n, Please open the file : Mat_Inv_out.txt located in the current folder. To see the output.\n\n");
//
//		FILE* fp;
//
//
//		fp = fopen("Mat_Inv_out", "w");
//
//		if (!fp)
//		{
//			fprintf(stderr, "Failed to open Mat_Inv_out.\n");
//			exit(1);
//		}
//
//
//
//		for (int qq = 0; qq < mybatch; qq++)
//		{
//
//			if (mybatch == 1)
//				fprintf(fp, "Input Matrix , M:\n\n");
//			else
//				fprintf(fp, "Input Matrix %d:\n\n", qq);
//
//
//
//
//			for (int i = 0; i < n; i++)
//			{
//				for (int j = 0; j < n; j++)
//				{
//					if (inputs[qq][i * n + j])
//						fprintf(fp, "%12f ", inputs[qq][i * n + j]);
//					else
//						fprintf(fp, "%11f ", inputs[qq][i * n + j]);
//				}
//
//				fprintf(fp, "\n");
//			}
//		}
//		fprintf(fp, "\n\n");
//
//		for (int qq = 0; qq < mybatch; qq++)
//		{
//			if (mybatch == 1)
//				fprintf(fp, "Inverse of the Input Matrix, Inv(M):\n\n");
//			else
//				fprintf(fp, "Inverse %d:\n\n", qq);
//			for (int i = 0; i < n; i++)
//			{
//				for (int j = 0; j < n; j++)
//				{
//					if (results[qq][i * n + j])
//						fprintf(fp, "%10f ", results[qq][i * n + j]);
//					else
//						fprintf(fp, "%9f ", results[qq][i * n + j]);
//
//				}
//
//				fprintf(fp, "\n");
//			}
//		}
//
//		fclose(fp);
//
//	}// end of if else condition for output
//
//	float* A, * B;
//
//	A = inputs[0];
//	B = results[0];
//	mul(A, B, n);
//
//	//mul(inputs[0][], results[0][], n );
//
//}
//
//////////////////////////////////////////////////////////////////////////
////
////
//////////////////////////////////////////////////////////////////////////
//
//int main(int argc, char* argv[])
//{
//	if (argc != 2)
//	{
//		printf("Usage: %s <matrix_width>\n", argv[0]);
//		return 0;
//	}
//
//	int w;
//	w = atoi(argv[1]);
//
//	test_invert(w);
//	return 0;
//}


//#include <stdio.h>
//#include <stdlib.h>
//#include <cublas_v2.h>
////#include<cublas.h>
//#include<math.h>
//#include<cuda_runtime.h>
//
//int main()
//{
//	cublasHandle_t handle;
//
//	cublasCreate(&handle);
//
//
//	int  size = 50; //矩阵的行和列
//	int num = 100;//矩阵组的矩阵个数
//	int* info;//用于记录LU分解是否成功
//	int* pivo;//用于记录LU分解的信息
//	cudaMalloc((void**)&info, sizeof(int) * num);
//	cudaMalloc((void**)&pivo, sizeof(int) * size * num);
//	float** mat = new float* [num];//待求逆的矩阵组
//	float** invMat = new float* [num];//存放逆矩阵的矩阵组
//	for (int i = 0; i < num; i++) {
//		cudaMalloc((void**)&mat[i], sizeof(float) * size * size);
//		cudaMalloc((void**)&invMat[i], sizeof(float) * size * size);
//		/*
//		这里将矩阵的数据载入mat[i]中,这里假设矩阵的数据在内存中是连续存放的
//		*/
//	}
//
//	float** gpuMat;
//	cudaMalloc((void**)&gpuMat, sizeof(float*) * num);
//	cudaMemcpy(gpuMat, mat, sizeof(float*) * num, cudaMemcpyHostToDevice);
//	//以上三步的目的是把host上的float ** 指针转变为 device上的 float ** 指针
//
//	cublasSgetrfBatched(handle, size, gpuMat, size, pivo, info, num);//第四个参数是矩阵的主导维，由于这里假设数据在内存中的存放是连续的，所以是size
//
//	
//	const float** constMat;
//	cudaMalloc((void**)&constMat, sizeof(float*) * num);
//	cudaMemcpy(constMat, gpuMat, sizeof(float*) * num, cudaMemcpyDeviceToDevice);
//	//以上三步的目的是把 float ** 指针转变为 float *[]指针
//
//	float** gpuInvMat;
//	cudaMalloc((void**)&gpuInvMat, sizeof(float*) * num);
//	cudaMemcpy(gpuInvMat, invMat, sizeof(float*) * num, cudaMemcpyHostToDevice);
//
//	//以上三步的目的是把host上的float ** 指针转变为 device上的 float ** 指针
//
//	cublasSgetriBatched(handle, size, constMat, size, pivo, gpuInvMat, size, info, num);
//
//	cudaFree(info);
//	cudaFree(pivo);
//	cudaFree(mat);
//	cudaFree(gpuMat);
//	cudaFree(gpuInvMat);
//	cudaFree(constMat);
//}



