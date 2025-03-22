/*
Author: Petrus E. Manurung
*/

#pragma once

#include <bitset>
#include <fstream>

#include "cx.h"
#include "TileCSR.h"

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

void generate_example_array(r_Ptr<int> data, int m = 16, int n = 16) {
    int rows[] = {0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 13, 13, 14, 14, 15};
    int cols[] = {0, 12, 14, 1, 2, 15, 1, 3, 12, 4, 5, 6, 7, 15, 13, 12, 13, 4, 6, 4, 7, 12, 7, 14, 4};
    // constexpr int len = std::size(rows);
    int len = 25;
    for(int i = 0; i < len; ++i) {
        data[rows[i] * m + cols[i]] = 1;
    }
}

template<typename T>
void printVector(cr_Ptr<T> data, int rows, int cols) {
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            if constexpr(std::is_integral_v<T>) std::printf("%d ", data[r * cols + c]);
            else if (std::is_floating_point_v<T>) std::printf("%f ", data[r * cols + c]);
        }
        std::printf("\n");
    }
}

template<typename T>
__device__ __forceinline__
int lowerBound(cr_Ptr<T> arr, T target, int len) {
    int l = 0;
    int r = len - 1;
    
    while(l <= r) {
        int m = (l + r) / 2;
        if(arr[m] == target) return m;
        else if(arr[m] < target) l = m + 1;
        else r = m - 1;
    }

    if(r < l) return r;
    return l;
}

template<typename T>
__host__
__device__ __forceinline__ 
int binarySearch(cr_Ptr<T> arr, T target, int len) {
    if(len == 0) return -1;
    int left = 0;
    int right = len - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target) {
        return mid;
        } else if (arr[mid] < target) {
        left = mid + 1;
        } else {
        right = mid - 1;
        }
    }

    return -1;
}

template<typename T>
concept ThrustVector = 
    std::is_same_v<T, thrust::device_vector<typename T::value_type, typename T::allocator_type>> ||
    std::is_same_v<T, thrust::host_vector<typename T::value_type, typename T::allocator_type>>;

struct getLow32 : thrust::unary_function<long long,int> {
    __host__ __device__
    int operator()(long long l) {
        return (int)l;
    }
};

struct getHigh32 : thrust::unary_function<long long,int> {
    __host__ __device__
    int operator()(long long l) {
        return *(reinterpret_cast<int*>(&l) + 1);
    }
};

struct is_not_neg1 {
  __host__ __device__
  bool operator()(int x) {
    return x != -1;
  }
};


template<typename ValueType, int tileSize>
void printInfo(std::ofstream &outfile, TileCSR_rev<ValueType,tileSize> const &tile, int tileNnz) noexcept
{
    outfile << "TileCSR:\n";
    outfile << "  mask:\n";
    for(int i = 0; i < tileSize; ++i) {
        std::bitset<16> val(tile.mask[i]);
        outfile << val.to_string().c_str() << " ";
        if((i + 1) % tileSize == 0 && i != tileSize - 1) {
            outfile << "\n";
        }
    }
    outfile << "\n";
    outfile << "  rowPtr:\n";
    for(int i = 0; i < tileSize; ++i) {
        outfile << static_cast<int>(tile.rowPtr[i]) << " ";
    }
    outfile << "\n";
    outfile << "  rowColIdx:\n";
    for(int i = 0; i < tileNnz; ++i) {
        std::bitset<8> val(tile.rowColIdx[i]);
        outfile << val.to_string().c_str() << " ";
        if((i + 1) % tileSize == 0 && i != tileNnz - 1) {
            outfile << "\n";
        }
    }
    outfile << "\n";
    outfile << "  vals:\n";
    for(int i = 0; i < tileNnz; ++i) {
        outfile << tile.vals[i] << " ";
        if((i + 1) % tileSize == 0 && i != tileNnz - 1) {
            outfile << "\n";
        }
    }
    outfile << "\n\n";
}

// template<typename ValueType, int tileSize>
// void printInfo(
//     std::ofstream &outfile, TileCSR_rev<ValueType,tileSize> const &tile, int tileNnz) noexcept
// {
//     outfile << "TileCSR:\n";
//     outfile << "  mask:\n";
//     for(int i = 0; i < tileSize; ++i) {
//         std::bitset<16> val(tile.mask[i]);
//         outfile << val.to_string().c_str() << " ";
//         if((i + 1) % tileSize == 0 && i != tileSize - 1) {
//             outfile << "\n";
//         }
//     }
//     outfile << "\n";
//     outfile << "  rowPtr:\n";
//     for(int i = 0; i < tileSize; ++i) {
//         outfile << static_cast<int>(tile.rowPtr[i]) << " ";
//     }
//     outfile << "\n";
//     outfile << "  rowColIdx:\n";
//     for(int i = 0; i < tileNnz; ++i) {
//         // std::bitset<8> val(tile.rowColIdx[i]);
//         // outfile << val.to_string().c_str() << " ";
//         auto val = tile.rowColIdx[i];
//         auto r = val >> 4;
//         auto c = val & 0xF;
//         outfile << r << "-" << c << " ";
//         if((i + 1) % tileSize == 0 && i != tileNnz - 1) {
//             outfile << "\n";
//         }
//     }
//     outfile << "\n";
//     outfile << "  vals:\n";
//     for(int i = 0; i < tileNnz; ++i) {
//         outfile << tile.vals[i] << " ";
//         if((i + 1) % tileSize == 0 && i != tileNnz - 1) {
//             outfile << "\n";
//         }
//     }
//     outfile << "\n\n";
// }

template<typename ValueType, int tileSize = 16>
void printInfo(
    std::ofstream &outfile, TileCSR_C_rev<ValueType,tileSize> const &tile, int tileNnz) noexcept
{
    outfile << "TileCSR_C_rev:\n";
    outfile << "  mask:\n";
    for(int i = 0; i < tileSize/2; ++i) {
        unsigned masq = tile.mask[i];
        std::bitset<16> val(masq >> 16);
        outfile << val.to_string().c_str() << "\n";
        val = masq;
        outfile << val.to_string().c_str() << "\n";
        if((i + 1) % tileSize == 0 && i != tileSize - 1) {
            outfile << "\n";
        }
    }
    outfile << "\n";
    outfile << "  rowPtr:\n";
    for(int i = 0; i < tileSize; ++i) {
        outfile << static_cast<int>(tile.rowPtr[i]) << " ";
    }
    outfile << "\n";
    // outfile << "  rowColIdx:\n";
    // for(int i = 0; i < tileNnz; ++i) {
    //     std::bitset<8> val(tile.rowColIdx[i]);
    //     outfile << val.to_string().c_str() << " ";
    //     if((i + 1) % tileSize == 0 && i != tileNnz - 1) {
    //         outfile << "\n";
    //     }
    // }
    // outfile << "\n";
    // outfile << "  vals:\n";
    // for(int i = 0; i < tileNnz; ++i) {
    //     outfile << tile.vals[i] << " ";
    //     if((i + 1) % tileSize == 0 && i != tileNnz - 1) {
    //         outfile << "\n";
    //     }
    // }
    // outfile << "\n\n";
}

template<typename ValueType, int tileSize>
void printInfo2(std::ofstream &outfile, TileCSR_C_rev<ValueType,tileSize> const &tile, int tileNnz) noexcept
{
    outfile << "TileCSR_C:\n";
    outfile << "  mask:\n";
    for(int i = 0; i < tileSize; ++i) {
        std::bitset<16> val(tile.mask[i]);
        outfile << val.to_string().c_str() << " ";
        if((i + 1) % tileSize == 0 && i != tileSize - 1) {
            outfile << "\n";
        }
    }
    outfile << "\n";
    outfile << "  vals:\n";
    for(int i = 0; i < tileNnz; ++i) {
        outfile << tile.vals[i] << " ";
        if((i + 1) % tileSize == 0 && i != tileNnz - 1) {
            outfile << "\n";
        }
    }
    outfile << "\n";
}

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