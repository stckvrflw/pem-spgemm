/*
Author: Petrus E. Manurung
*/

#pragma once

#include <bitset>
#include <fstream>

#define EXIT_FAILURE 1

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_THRUST(func)                                                     \
{                                                                              \
    auto result = (func);                                                      \
    if (result != thrust::success) {                                           \
        printf("THrust API failed at line %d with error: %s (%d)\n",           \
               __LINE__, thrust::detail::get_error_message(result).c_str(),    \
               result);                                                        \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_THRUST_ERROR(function) \
  do { \
    try { \
      (function); \
    } \
    catch (const std::bad_alloc& e) { \
       size_t free_mem_bytes, total_mem_bytes; \
       cudaMemGetInfo(&free_mem_bytes, &total_mem_bytes); \
       std::cout << "thrust::bad_alloc of Thrust GPU memory: Free " << (double)free_mem_bytes  / (1024 * 1024 * 1024) << " / Total " << (double)total_mem_bytes / (1024 * 1024 * 1024) << "GB: " << e.what() << " [Line: " << __LINE__ << ", " << __FUNCTION__ << ", " << __FILE__ << std::endl; \
       cudaGetLastError(); \
       return false; \
    } \
    catch (const thrust::system_error& e) { \
       std::cout << "thrust::system_error of Thrust GPU memory: " << e.what() << " [Line: " << __LINE__ << ", " << __FUNCTION__ << ", " << __FILE__ << std::endl; \
       cudaGetLastError(); \
       return false; \
    } \
  } while (0)


// template<typename T>
// __device__ __forceinline__
// int lowerBound(cr_Ptr<T> arr, T target, int len) {
//     int l = 0;
//     int r = len - 1;
    
//     while(l <= r) {
//         int m = (l + r) / 2;
//         if(arr[m] == target) return m;
//         else if(arr[m] < target) l = m + 1;
//         else r = m - 1;
//     }

//     if(r < l) return r;
//     return l;
// }

template<typename T>
__host__ __device__ __forceinline__
int lowerBound(T const *__restrict__ arr, T target, int len) {
    int l = 0;
    int r = len - 1;

    int ans = len;
    
    while(l <= r) {
        int m = (l+r)/2;
        if(arr[m]>=target)
        {
            ans = m;
            r = m-1;
        }
        else l = m+1;
    }

    return ans;
}


// template<typename T>
__host__
__device__ __forceinline__ 
int binarySearch(int const *__restrict__ arr, int target, int len) {
    // auto custom_load = [](uintptr_t global, int &val) __attribute__((always_inline)) { asm ("ld.global.cv.s32 %0, [%1];" : "=r"(val) : "l"(global));};
    
    if(len == 0) return -1;
    int left = 0;
    int right = len - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        // int val;
        // custom_load((uintptr_t)&arr[mid], val);

        int val = arr[mid];

        if (val == target) {
        return mid;
        } else if (val < target) {
        left = mid + 1;
        } else {
        right = mid - 1;
        }
    }

    return -1;
}

struct getLow32 : thrust::unary_function<long long,int> {
    __host__ __device__
    int operator()(long long l) {
        // return (int)l;
        return (l & 0x0FFFFFFFF);
    }
};

struct getHigh32 : thrust::unary_function<long long,int> {
    __host__ __device__
    int operator()(long long l) {
        // return *(reinterpret_cast<int*>(&l) + 1);
        return (l >> 32);
    }
};

struct swap32 : thrust::unary_function<long long, long long>{
    __host__ __device__
    long long operator()(long long l){
        return (static_cast<long long>(l << 32) | (l >> 32));
    }
};

struct is_not_neg1 {
  __host__ __device__
  bool operator()(int x) {
    return x != -1;
  }
};

template<int tileSize = 16>
__device__
__forceinline__
void __tile16x16_transpose_sync(auto &warp, auto *tile) 
{
    typename std::remove_pointer<decltype(tile)>::type my_data[2][4];

    int my_data_start[2];
    my_data_start[0] = warp.thread_rank() / 4 * (tileSize) + warp.thread_rank() * (tileSize / 4);
    my_data_start[1] = my_data_start[0] + 2;

    #pragma unroll
    for(int n = 0; n < 2; ++n) {
        int my_data_start_y = my_data_start[n] / tileSize;
        int my_data_start_x = my_data_start[n] % tileSize;
        
        
        my_data[n][0] = tile[my_data_start_y * tileSize + my_data_start_x + 0];
        my_data[n][1] = tile[my_data_start_y * tileSize + my_data_start_x + 1]; 
        my_data[n][2] = tile[(my_data_start_y + 1) * tileSize + my_data_start_x + 0]; 
        my_data[n][3] = tile[(my_data_start_y + 1) * tileSize + my_data_start_x + 1]; 
    }

    #pragma unroll
    for(int n = 0; n < 2; ++n) {
        int my_data_start_y = my_data_start[n] / tileSize;
        int my_data_start_x = my_data_start[n] % tileSize;
        
        
        tile[my_data_start_x * tileSize + my_data_start_y + 0] = my_data[n][0];
        tile[my_data_start_x * tileSize + my_data_start_y + 1] = my_data[n][2];
        tile[(my_data_start_x + 1) * tileSize + my_data_start_y + 0] = my_data[n][1];
        tile[(my_data_start_x + 1) * tileSize + my_data_start_y + 1] = my_data[n][3];
    }

    warp.sync();
}