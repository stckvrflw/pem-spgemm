/*
Author: Petrus E. Manurung
July 2025
*/

#include <cstdlib>
#include <cstdio>

#include <fstream>
#include <thread>
#include <algorithm>
#include <chrono>
#include <vector>
#include <complex>
#include <regex>
#include <type_traits>

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cooperative_groups/reduce.h>

#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/unique.h>
#include <thrust/partition.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>
#include <thrust/host_vector.h>

#include <cub/block/block_scan.cuh>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/device_vector.hpp>

#include "fast_matrix_market/fast_matrix_market.hpp"
#include "utilities.h"
#include "spgemm_nsparse_kernel.h"

namespace cg = cooperative_groups;

template<typename ValueType>
void read_matrix_market
(
    char const *matrix_path,
    int **I, 
    int **J, 
    ValueType **vals, 
    int &rows, int &cols, int &nnz,
    bool &isSymmetric
)
{
    std::ifstream file(matrix_path);
    std::vector<int> _I;
    std::vector<int> _J;
    std::vector<ValueType> _vals;
    std::vector<std::complex<double>> _complex_vals;
    int _rows, _cols, _nnz;
    fast_matrix_market::matrix_market_header mtx_header;
    fast_matrix_market::read_header(file, mtx_header);
    file.clear();
    file.seekg(0);

    if(mtx_header.symmetry == fast_matrix_market::symmetry_type::symmetric)
    {
        isSymmetric = true;
    }

    if(mtx_header.field == fast_matrix_market::field_type::complex)
    {
    fast_matrix_market::read_matrix_market_triplet(file,
        _rows, _cols,
        _I, _J, _complex_vals);
    _nnz = _complex_vals.size();
    }
    else
    {
    fast_matrix_market::read_matrix_market_triplet(file,
                                                    _rows, _cols,
                                                    _I, _J, _vals);
    _nnz = _vals.size();
    }
    
    file.close();


    cudaMallocHost(I, sizeof(remove_all_pointers_t<decltype(I)>) * _nnz);
    cudaMallocHost(J, sizeof(remove_all_pointers_t<decltype(J)>) * _nnz);
    cudaMallocHost(vals, sizeof(remove_all_pointers_t<decltype(vals)>) * _nnz);

    rows = _rows;
    cols = _cols;
    nnz = _nnz;

    std::copy(_I.begin(), _I.end(), *I);
    std::copy(_J.begin(), _J.end(), *J);

    if(mtx_header.field == fast_matrix_market::field_type::complex)
    {
        int i = 0;
        for(auto cval: _complex_vals)
        {
            ValueType temp = reinterpret_cast<ValueType(&)[2]>(cval)[0]; // get real with complex
            (*vals)[i++] = temp;
        }
    }
    else
    std::copy(_vals.begin(), _vals.end(), *vals);
}

template<unsigned tileSize = 16>
__global__ void __launch_bounds__(tileSize * tileSize)
decide_which_tile
(
    long long *__restrict__ participating_tiles,
    int const *__restrict__ d_I,
    int const *__restrict__ d_J,
    int nnz
)
{
    auto grid = cg::this_grid();
    auto my_tid = grid.thread_rank();
    // out of range check
    if(my_tid >= nnz) return;
    int my_y = d_I[my_tid];
    int my_x = d_J[my_tid];
    int this_tile_y = my_y >> 4;
    int this_tile_x = my_x >> 4;

    long long my_tile = this_tile_y;
    my_tile <<= 32;
    my_tile |= this_tile_x;
    participating_tiles[my_tid] = my_tile;
}

template<typename ValueType, int tileSize = 16>
__global__ void __launch_bounds__(tileSize * tileSize)
generate_tiles_csr
(
    ValueType *__restrict__ d_tiles_vals,
    uint16_t *__restrict__ d_tiles_masks,
    uint8_t *__restrict__ d_tiles_rowPtr,
    uint8_t *__restrict__ d_tiles_rowColIdx,
    long long const *__restrict__ participating_tiles,
    int participating_tiles_size,
    int *__restrict__ perTileNnz,
    int const *__restrict__ d_J,
    ValueType const *__restrict__ d_vals,
    int const *__restrict__ d_rowPtr,
    int d_rowPtr_size
)
{
    using MaskType = uint16_t;
    using IdxType = uint8_t;

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    auto tid = block.thread_rank();

    unsigned block_id = grid.block_rank();
    while(block_id < participating_tiles_size) 
    {
        long long block_tile = participating_tiles[block_id];
        int block_tile_x = *(reinterpret_cast<int*>(&block_tile));
        int block_tile_y = *(reinterpret_cast<int*>(&block_tile)+1);

        int block_tile_offset_x = block_tile_x << 4;
        int block_tile_offset_y = block_tile_y << 4;
        int block_d_tiles_offset = block_id;

        ValueType thread_val {};
        int thread_J = -1;

        __shared__ int temp_buffer[16];

        auto my_row_group = cg::tiled_partition<tileSize>(block);
        
        int my_row_group_rowPtr_offset = block_tile_offset_y + my_row_group.meta_group_rank();
        if(my_row_group_rowPtr_offset < d_rowPtr_size)
        {
            int my_row_group_rowPtr = d_rowPtr[my_row_group_rowPtr_offset];
            int my_row_group_rowSize = d_rowPtr[my_row_group_rowPtr_offset + 1] - d_rowPtr[my_row_group_rowPtr_offset];

            int thread_offset = binarySearch(&d_J[my_row_group_rowPtr], block_tile_offset_x + (int)my_row_group.thread_rank(), my_row_group_rowSize);
            if(thread_offset != -1){
                thread_offset += my_row_group_rowPtr;
                thread_J = d_J[thread_offset];
                thread_val = d_vals[thread_offset];
            }
        }
        my_row_group.sync();

        IdxType my_RowColIdx = (static_cast<IdxType>(my_row_group.meta_group_rank()) << 4) | (thread_J%16);
        unsigned my_row_mask = my_row_group.ballot(thread_J != -1);
        unsigned my_row_nnz = __popc(my_row_mask);

        if(my_row_group.thread_rank() == 0) {
            d_tiles_masks[(block_d_tiles_offset<<4)+my_row_group.meta_group_rank()] = my_row_mask;
            temp_buffer[my_row_group.meta_group_rank()] = my_row_nnz;
        }
        block.sync();

        if(my_row_group.meta_group_rank() == 0) {
            my_row_nnz = temp_buffer[my_row_group.thread_rank()];
            my_row_nnz = cg::exclusive_scan(my_row_group, my_row_nnz);
            d_tiles_rowPtr[(block_d_tiles_offset<<4)+my_row_group.thread_rank()] = my_row_nnz;
        }

        int tile_offset = perTileNnz[block_id];

        int my_loc = (thread_J != -1) ? 1 : 0;
        using BlockScan = cub::BlockScan<int, 256>;
        __shared__ typename BlockScan::TempStorage temp_storage;
        BlockScan(temp_storage).ExclusiveSum(my_loc, my_loc);

        if(thread_J != -1)
        {
            d_tiles_vals[tile_offset + my_loc] = thread_val;
            d_tiles_rowColIdx[tile_offset + my_loc] = my_RowColIdx;
        }

        block_id += grid.num_blocks();
    }
}

__global__ void __launch_bounds__(256)
__transpose_B_mask
(
    uint16_t *__restrict__ Btiles_transposed_mask,
    uint16_t const *__restrict__ Btiles_mask,
    int Btiles_size
)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint16_t thread_buf[16];
    while(idx < Btiles_size)
    {
        uint16_t tile_mask[16];
        *(reinterpret_cast<ulonglong4*>(&tile_mask[0])) = *(reinterpret_cast<ulonglong4 const*>(&Btiles_mask[(idx<<4)]));
        
        #pragma unroll
        for(int m = 0; m < 16; ++m)
        {
            uint16_t temp = 0;
            #pragma unroll
            for(int i = 0; i < 16; ++i)
            {
                temp |= (((tile_mask[i] >> m) & 1) << i);
            }
            thread_buf[m] = temp;
        }
        *(reinterpret_cast<ulonglong4*>(&Btiles_transposed_mask[(idx<<4)])) = *(reinterpret_cast<ulonglong4*>(thread_buf));

        idx += (gridDim.x << 8);
    }
}

// from TileSpGEMM
__forceinline__ __device__ int sum_32_shfl(int sum)
{
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
}

// from TileSpGEMM
__global__ void tile_spgemm_step1_cuda_spa_kernel(int *d_blkrowptrA, int *d_blkcolidxA, int blkmA,
                                                  int *d_blkrowptrB, int *d_blkcolidxB, int blknB,
                                                  int *d_blkrowptrC)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;
    __shared__ unsigned int bitmask[WARP_PER_BLOCK * SPA_INT_PER_WARP];

    if (global_warp_id >= blkmA)
        return;

    const int nmasks = ceil((float)blknB / (float)32);
    const int local_warp_id = threadIdx.x >> 5; //global_id / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    unsigned int *bitmask_local = &bitmask[local_warp_id * SPA_INT_PER_WARP];

    for (int i = lane_id; i < nmasks; i += WARP_SIZE)
        bitmask_local[i] = 0;

    int astart = d_blkrowptrA[global_warp_id];
    int astop = d_blkrowptrA[global_warp_id + 1];
    for (int i = astart; i < astop; i++)
    {
        int rowidx = d_blkcolidxA[i];
        int bstart = d_blkrowptrB[rowidx];
        int bstop = d_blkrowptrB[rowidx + 1];
        for (int j = bstart + lane_id; j < bstop; j += WARP_SIZE)
        {
            int colidx = d_blkcolidxB[j];
            unsigned int mask = 1 << (31 - colidx % 32);
            atomicOr(&bitmask_local[colidx / 32], mask);
        }
    }
    //__syncthreads();

    int cnt = 0;
    for (int i = lane_id; i < nmasks; i += WARP_SIZE)
        cnt += __popc(bitmask_local[i]);
    cnt = sum_32_shfl(cnt);

    if (!lane_id)
        d_blkrowptrC[global_warp_id] = cnt;
}

// from TileSpGEMM
__global__ void tile_spgemm_step1_numeric_cuda_spa_kernel(int *d_blkrowptrA, int *d_blkcolidxA, int blkmA,
                                                          int *d_blkrowptrB, int *d_blkcolidxB, int blknB,
                                                          int *d_blkrowptrC, int *d_blkrowidxC, int *d_blkcolidxC,
                                                          int *d_spec_intersection_cnt, int *d_spec_intersection_posa, int *d_spec_intersection_posb)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;
    __shared__ unsigned int bitmask[WARP_PER_BLOCK * SPA_INT_PER_WARP];

    if (global_warp_id >= blkmA)
        return;

    const int nmasks = ceil((float)blknB / (float)32);
    const int nmasks_warpwise = ceil((float)nmasks / (float)WARP_SIZE) * WARP_SIZE; // make sure shfl func works
    const int local_warp_id = threadIdx.x >> 5;                                     //global_id / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    unsigned int *bitmask_local = &bitmask[local_warp_id * SPA_INT_PER_WARP];

    for (int i = lane_id; i < nmasks_warpwise; i += WARP_SIZE)
        bitmask_local[i] = 0;

    int cbase = d_blkrowptrC[global_warp_id];

    int astart = d_blkrowptrA[global_warp_id];
    int astop = d_blkrowptrA[global_warp_id + 1];
    for (int i = astart; i < astop; i++)
    {
        int rowidx = d_blkcolidxA[i];
        int bstart = d_blkrowptrB[rowidx];
        int bstop = d_blkrowptrB[rowidx + 1];
        for (int j = bstart + lane_id; j < bstop; j += WARP_SIZE)
        {
            int colidx = d_blkcolidxB[j];
            unsigned int mask = 1 << (31 - colidx % 32);
            atomicOr(&bitmask_local[colidx / 32], mask);
        }
    }

    int cnt = 0;
    int offset = 0;
    for (int i = lane_id; i < nmasks_warpwise; i += WARP_SIZE)
    {
        unsigned int maski = bitmask_local[i];
        int cnt = __popc(maski);

        // inclusive scan
        int cnt_scan = scan_32_shfl(cnt, lane_id);
        cnt_scan += offset;

        // sum
        offset = __shfl_sync(0xffffffff, cnt_scan, 31);

        // to exclusive scan
        cnt_scan -= cnt;

        // write to gmem
        int localoff = 0;
#pragma unroll
        for (int biti = 0; biti < 32; biti++)
        {
            if ((maski >> (31 - biti)) & 0x1)
            {
                d_blkrowidxC[cbase + cnt_scan + localoff] = global_warp_id;
                d_blkcolidxC[cbase + cnt_scan + localoff] = i * 32 + biti;
                localoff++;
            }
        }
    }
}


template<int pass>
__device__ __forceinline__
// we iterate on lane_iter, search on lane_targ
int __find_pairs
(
    auto &warp,
    int *__restrict__ pairs_a,
    int *__restrict__ pairs_b,
    int const *__restrict__ lane_iter,
    int lane_iter_len,
    int const *__restrict__ lane_targ,
    int const lane_targ_len,
    int const iter_offset,
    int const targ_offset,
    int const AorB,
    int const *__restrict__ B_tileOffsets,
    int &insertion_start
)
{
    int __local_count = 0; // for use when pass = 0 only
    for(int i = warp.thread_rank(); i < lane_iter_len; i+=32) {
        auto loop_participants = cg::coalesced_threads();
        int found = binarySearch(lane_targ, lane_iter[i], lane_targ_len);

// guard to Independent Thread Scheduling
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
        if constexpr(pass == 1)
        {
        loop_participants.sync();
        }
#endif
#endif
        if(found != -1) {
            if constexpr(pass == 1) // 1 = second pass
            {   
            auto lucky_ones = cg::coalesced_threads();
            int first = iter_offset + i;
            int second = targ_offset + found;
            if(!AorB) // meaning its B, need to swap-- order is AB
            std::swap(first, second);
            asm ("ld.global.cv.s32 %0, [%1];" : "=r"(second) : "l"(&B_tileOffsets[second]));
            int targ_idx = insertion_start + lucky_ones.thread_rank();
            __stcs(&pairs_a[targ_idx], first);
            __stcs(&pairs_b[targ_idx], second);
            if(lucky_ones.thread_rank() == 0) insertion_start += lucky_ones.num_threads();
            }
            else // 0 = first pass, only count how many are there
            ++__local_count;
        }
    }
    return __local_count;
}

template<int pass>
__global__ void __launch_bounds__(256) 
pem_spgemm_step2_search_pairs
(
    int *__restrict__ pairs_a,
    int *__restrict__ pairs_b,
    int const *__restrict__ C_colIdx,
    int const *__restrict__ A_rowPtr,
    int const *__restrict__ A_colIdx,
    int const *__restrict__ B_colPtr,
    int const *__restrict__ B_rowIdx,
    int C_rowPtr_size,
    int C_colIdx_size,
    int const *__restrict__ B_tileOffsets,
    int const *__restrict__ C_rowIdx,
    int *__restrict__ pairs_insertion_offset
)
{
    auto wt_s32 = []<typename T>(uintptr_t global, T &val) __attribute__((always_inline)) { asm volatile("st.global.cs.s32 [%0], %1;" : : "l"(global) , "r"(val)); };

    __shared__ int warp_insertion_start[8];

    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int w_start = (blockIdx.x << 3) + (threadIdx.x >> 5);
    while(w_start < C_colIdx_size) 
    {
        int const w_C_col = __ldcv(&C_colIdx[w_start]);
        int const w_C_row = __ldcv(&C_rowIdx[w_start]);
        decltype(A_colIdx) A_colIdx_segment = A_colIdx + A_rowPtr[w_C_row];
        decltype(B_rowIdx) B_rowIdx_segment = B_rowIdx + B_colPtr[w_C_col];
        int const A_colIdx_segment_len = A_rowPtr[w_C_row + 1] - A_rowPtr[w_C_row];
        int const B_rowIdx_segment_len = B_colPtr[w_C_col + 1] - B_colPtr[w_C_col];
        int const AorB = A_colIdx_segment_len <= B_rowIdx_segment_len ? 1 : 0; // A = 1; B = 0;

        if constexpr(pass == 0) 
        {
        int curr_count = 0;
        if(AorB)
        curr_count = __find_pairs<0>(warp, nullptr, nullptr, A_colIdx_segment, A_colIdx_segment_len, B_rowIdx_segment, B_rowIdx_segment_len, A_rowPtr[w_C_row], B_colPtr[w_C_col], AorB, B_tileOffsets, warp_insertion_start[warp.meta_group_rank()]);
        else
        curr_count = __find_pairs<0>(warp, nullptr, nullptr, B_rowIdx_segment, B_rowIdx_segment_len, A_colIdx_segment, A_colIdx_segment_len, B_colPtr[w_C_col], A_rowPtr[w_C_row], AorB, B_tileOffsets, warp_insertion_start[warp.meta_group_rank()]);
        
        curr_count = cg::reduce(warp, curr_count, cg::plus<int>());
        cg::invoke_one(warp, wt_s32, (uintptr_t)&pairs_insertion_offset[w_start], curr_count);
        }
        else 
        {
        if(warp.thread_rank() == 0) warp_insertion_start[warp.meta_group_rank()] = pairs_insertion_offset[w_start];
        if(AorB)
        __find_pairs<1>(warp ,pairs_a, pairs_b, A_colIdx_segment, A_colIdx_segment_len, B_rowIdx_segment, B_rowIdx_segment_len, A_rowPtr[w_C_row], B_colPtr[w_C_col], AorB, B_tileOffsets, warp_insertion_start[warp.meta_group_rank()]);
        else
        __find_pairs<1>(warp, pairs_a, pairs_b, B_rowIdx_segment, B_rowIdx_segment_len, A_colIdx_segment, A_colIdx_segment_len, B_colPtr[w_C_col], A_rowPtr[w_C_row], AorB, B_tileOffsets, warp_insertion_start[warp.meta_group_rank()]);
        }

        w_start += (gridDim.x << 3);
    }
}

template<int tileSize = 16>
__global__ void __launch_bounds__(4 * 32) 
pem_spgemm_step2_compute_CMasksAndOffsets
(
    int const *__restrict__ d_pairs_a,
    int const *__restrict__ d_pairs_b,
    int Ctiles_size,
    int *__restrict__ _C_perTileNnz,
    uint16_t const *__restrict__ Atiles_masks,
    uint16_t const *__restrict__ Btiles_masks,
    uint16_t const *__restrict__ Btiles_transposed_mask,
    int const *__restrict__ pairs_insertion_offset,
    unsigned *__restrict__ Ctiles_mask
)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    auto isZero = [](unsigned n) __attribute__((always_inline)) { return ((n | (~n + 1)) >> 31) & 1; };
    auto wt_u32 = []<typename T>(uint64_t global, T val) __attribute__((always_inline)) { asm volatile("st.global.cs.u32 [%0], %1;" : : "l"(global) , "r"(val)); };

    int const quarter_mgr = threadIdx.x >> 3;
    int const quarter_tr = threadIdx.x % 8;
    int quarter_group_C_idx = (blockIdx.x << 4) + (threadIdx.x >> 3);
    auto quarter_group = cg::tiled_partition<8>(warp);
    while(quarter_group_C_idx < Ctiles_size)
    {
        unsigned quarter_tr_Ctiles_mask = 0;
        for(int pair = __ldcg(&pairs_insertion_offset[quarter_group_C_idx]); pair < __ldcg(&pairs_insertion_offset[quarter_group_C_idx+1]); ++pair)
        {
            int const quarter_local_group_tile_idx_A = __ldcs(&d_pairs_a[pair]);
            int const quarter_local_group_tile_idx_B = __ldcs(&d_pairs_b[pair]);

            unsigned C_mask = 0;
            #pragma unroll
            for(int n = 0; n < tileSize; ++n) C_mask |= (isZero((Atiles_masks[(quarter_local_group_tile_idx_A<<4) + (quarter_tr<<1)] & Btiles_transposed_mask[(quarter_local_group_tile_idx_B<<4) + n])) << n);
            C_mask <<= 16;
            #pragma unroll
            for(int n = 0; n < tileSize; ++n) C_mask |= (isZero((Atiles_masks[(quarter_local_group_tile_idx_A<<4) + ((quarter_tr<<1)+1)] & Btiles_transposed_mask[(quarter_local_group_tile_idx_B<<4) +n])) << n);

            quarter_tr_Ctiles_mask |= C_mask;
        }

        wt_u32((uint64_t)&Ctiles_mask[(quarter_group_C_idx<<3)+quarter_tr], quarter_tr_Ctiles_mask);

        int tileC_nnz = cg::reduce(quarter_group, __popc(quarter_tr_Ctiles_mask), cg::plus<int>());
        cg::invoke_one(quarter_group, wt_u32, (uint64_t)(_C_perTileNnz + quarter_group_C_idx), tileC_nnz);

        quarter_group_C_idx += (gridDim.x << 4);
    }
}

template<int tileSize = 16>
__global__ void __launch_bounds__(4 * 32) 
pem_spgemm_step2_compute_CrowColIdx
(
    int Ctiles_size,
    int *__restrict__ _C_perTileNnz,
    uint8_t *__restrict__ Ctiles_rowColIdx,
    uint8_t *__restrict__ Ctiles_rowPtr,
    unsigned const *__restrict__ Ctiles_mask
)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    auto local_group = cg::tiled_partition<16>(warp);
    int const lgmgr = local_group.meta_group_rank();
    int const lgtr = local_group.thread_rank();

    int local_group_Ctiles_idx_start = (grid.block_rank() << 3) + (warp.meta_group_rank() << 1) + lgmgr;
    while(local_group_Ctiles_idx_start < Ctiles_size) 
    {
        unsigned my_Cmask = __ldcs(&Ctiles_mask[(local_group_Ctiles_idx_start<<3) + (lgtr>>1)]);
        my_Cmask >>= (((lgtr % 2) ^ 0x1) << 4);
        my_Cmask &= 0xFFFF;
        int const Ctile_offset = __ldcs(&_C_perTileNnz[local_group_Ctiles_idx_start]);

        int my_offset = cg::exclusive_scan(local_group, __popc(my_Cmask));
        __stcs(&Ctiles_rowPtr[(local_group_Ctiles_idx_start<<4)+lgtr], my_offset);

        for(int n = 1; n <= __popc(my_Cmask); ++n)
        {
            unsigned c = __fns(my_Cmask, 0, n);
            unsigned my_rowColIdx = (lgtr << 4) | c;
            __stcs(&Ctiles_rowColIdx[Ctile_offset + (my_offset++)], my_rowColIdx);
        }

        local_group_Ctiles_idx_start += (grid.num_blocks() << 3);
    }
}

template<typename ValueType, int tileSize = 16>
__global__ void 
__launch_bounds__(tileSize * tileSize / 2)
pem_spgemm_step3_accumulate
(
    int const *__restrict__ d_pairs_a,
    int const *__restrict__ d_pairs_b,
    int Ctiles_size,
    ValueType *__restrict__ Ctiles_vals,
    int *__restrict__ _C_perTileNnz,
    uint8_t const *__restrict__ Ctiles_rowColIdx,
    int const *__restrict__ A_perTileNnz,
    ValueType const *__restrict__ Atiles_vals,
    uint16_t const *__restrict__ Atiles_masks,
    uint8_t const *__restrict__ Atiles_rowPtr,
    int const *__restrict__ B_perTileNnz,
    ValueType const *__restrict__ Btiles_vals,
    uint16_t const *__restrict__ Btiles_masks,
    uint8_t const *__restrict__ Btiles_rowPtr,
    uint16_t const *__restrict__ Btiles_transposed_mask,
    int const *__restrict__ pairs_insertion_offset
)
{
    using IdxType = uint8_t;

    __shared__ IdxType warp_tileC_rowColIdx[4][256];

    int const warp_tr = threadIdx.x % 32;
    int const warp_mgr = threadIdx.x >> 5;

    int warp_tileC_idx = (blockIdx.x << 2) + warp_mgr;
    while(warp_tileC_idx < Ctiles_size)
    {
        int const tileC_offset = _C_perTileNnz[warp_tileC_idx];
        int const tileC_nnz = _C_perTileNnz[warp_tileC_idx + 1] - tileC_offset;
        int const d_pairs_count = pairs_insertion_offset[warp_tileC_idx+1] - pairs_insertion_offset[warp_tileC_idx];

        for(int i = warp_tr; i < tileC_nnz; i += 32)
        {
            warp_tileC_rowColIdx[warp_mgr][i] = __ldcs(&Ctiles_rowColIdx[tileC_offset+i]);
        }

        for(int pair = pairs_insertion_offset[warp_tileC_idx]; pair < pairs_insertion_offset[warp_tileC_idx] + d_pairs_count; ++pair) 
        {
            int const A = d_pairs_a[pair];
            int const B = d_pairs_b[pair];
            int const A_global_offset = A_perTileNnz[A];
            int const B_global_offset = B_perTileNnz[B];

            // calculate C
            for(int n = warp_tr; n < tileC_nnz; n+=32)
            {
                int const r = warp_tileC_rowColIdx[warp_mgr][n] >> 4;
                int const c = warp_tileC_rowColIdx[warp_mgr][n] & 0xF;
                unsigned lane_mask = Atiles_masks[(A<<4)+r] & Btiles_transposed_mask[(B<<4)+c];
                while(lane_mask)
                {
                    int const ffs = __ffs(lane_mask)-1;
                    int const A_offset = __popc( Atiles_masks[(A<<4)+r] & (0xFFFFU >> (16-ffs)) );
                    int const B_offset = __popc( Btiles_masks[(B<<4)+ffs] & (0xFFFFU >> (16-c)) );
                    Ctiles_vals[tileC_offset + n] += Atiles_vals[A_global_offset+Atiles_rowPtr[(A<<4)+r]+A_offset] * Btiles_vals[B_global_offset+Btiles_rowPtr[(B<<4)+ffs]+B_offset];

                    lane_mask &= (~(1 << (ffs)));
                }
            }
        }
        warp_tileC_idx += (gridDim.x << 2);
    }
}

template<typename ValueType, int tileSize = 16>
__global__ void __launch_bounds__(tileSize * tileSize)
sanitize_C
(
    int *__restrict__ rows, 
    int *__restrict__ cols, 
    uint8_t const *__restrict__ Ctiles_rowColIdx,
    ValueType const *__restrict__ Ctiles_vals,
    int Ctiles_size,
    int const *__restrict__ _C_tileRowIdx,
    int const *__restrict__ _C_tileColIdx, 
    int const *__restrict__ _C_perTile_Nnz
)
{
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());

    int warp_Ctiles_id = (blockIdx.x << 3) + warp.meta_group_rank();
    while(warp_Ctiles_id < Ctiles_size)
    {
        int warp_Ctile_x = _C_tileColIdx[warp_Ctiles_id];
        int warp_Ctile_y = _C_tileRowIdx[warp_Ctiles_id];
        int warp_Ctile_offset = _C_perTile_Nnz[warp_Ctiles_id];
        for(int n = warp.thread_rank(); n < _C_perTile_Nnz[warp_Ctiles_id+1]-_C_perTile_Nnz[warp_Ctiles_id]; n+=warp.num_threads())
        {
            int idx = warp_Ctile_offset + n;
            uint8_t t_rowColIdx = Ctiles_rowColIdx[warp_Ctile_offset + n];
            rows[idx] = (warp_Ctile_y<<4) + (t_rowColIdx>>4);
            cols[idx] = (warp_Ctile_x<<4) + (t_rowColIdx&0xF);
        }

        warp_Ctiles_id += (gridDim.x << 3);
    }
}

#define STREAM_A streams[STREAMA]
#define STREAM_B streams[STREAMB]
#define STREAM_C streams[STREAMC]
#define STREAM_D streams[STREAMD]
#define STREAM_E streams[STREAME]

enum STREAMS {
    STREAMA,
    STREAMB,
    STREAMC,
    STREAMD,
    STREAME,
    STREAMS_COUNT,
};

#ifndef REPEAT
#define REPEAT 1
#endif

int main(int argc, char *argv[]) 
{
    if(argc <= 1 || argc > 4) {
        std::cout << "Provide a matrix market file path. Exiting.\n";
        exit(1);
    }

    constexpr int tileSize = 16;
    using ValueType = double;

    cudaEvent_t A_tileConversion_start, A_tileConversion_end, B_tileConversion_start, B_tileConversion_end;
    cudaEvent_t high_level_multiply_start, high_level_multiply_end;
    cudaEvent_t cusparse_start, cusparse_end;
    cudaEvent_t allocate_c_start, sp0_end, sp1_start, sp1_end, aC_start, aC_end, setOffset_start, allocate_c_end;
    cudaEvent_t accumulator_end;
    cudaEvent_t sanitize_C_start, sanitize_C_end;

    cudaEventCreate(&A_tileConversion_start);
    cudaEventCreate(&A_tileConversion_end);
    cudaEventCreate(&B_tileConversion_start);
    cudaEventCreate(&B_tileConversion_end);
    cudaEventCreate(&high_level_multiply_start);
    cudaEventCreate(&high_level_multiply_end);
    cudaEventCreate(&cusparse_start);
    cudaEventCreate(&cusparse_end);
    cudaEventCreate(&allocate_c_start);
    cudaEventCreate(&sp0_end);
    cudaEventCreate(&sp1_start);
    cudaEventCreate(&sp1_end);
    cudaEventCreate(&aC_start);
    cudaEventCreate(&aC_end);
    cudaEventCreate(&setOffset_start);
    cudaEventCreate(&allocate_c_end);
    cudaEventCreate(&accumulator_end);
    cudaEventCreate(&sanitize_C_start);
    cudaEventCreate(&sanitize_C_end);

    std::array<cudaStream_t, STREAMS_COUNT> streams;
    std::for_each(streams.begin(), streams.end(), [](cudaStream_t &s){cudaStreamCreate(&s);});

    auto tileCSR_conversion_start = std::chrono::high_resolution_clock::now();

    // HOST MATRIX A ----------------------
    int *A_I, *A_J;
    ValueType *A_val;
    int A_rows, A_cols, A_nnz;
    bool A_isSymmetric = false;
    //-------------------------------------

    // HOST MATRIX B ----------------------
    int *B_I, *B_J;
    ValueType *B_val;
    int B_rows, B_cols, B_nnz;
    bool B_isSymmetric = false;
    //-------------------------------------

    unsigned long long flop = 0;

    std::jthread read_A(read_matrix_market<ValueType>, std::ref(argv[1]), &A_I, &A_J, &A_val, std::ref(A_rows), std::ref(A_cols), std::ref(A_nnz), std::ref(A_isSymmetric));
    read_matrix_market(argv[1], &B_I, &B_J, &B_val, B_rows, B_cols, B_nnz, B_isSymmetric);
    read_A.join();

    if(A_rows != A_cols && argc < 4)
    {
        std::cout << "input is rectangular. Only AAt is possible. Exiting.\n";
        exit(1);
    }

    if(argc == 4 && argv[3])
    {
        std::swap(B_I, B_J);
        std::swap(B_rows, B_cols);
    }

    std::cout 
    << "MATRIX A\n"
    << "filepath: " << argv[1] << "\n"
    << "Rows: " << A_rows << "\n"
    << "Cols: " << A_cols << "\n"
    << "Nnz: " << A_nnz << "\n";

    std::cout 
    << "MATRIX B\n"
    << "filepath: " << argv[1] << "\n"
    << "Rows: " << B_rows << "\n"
    << "Cols: " << B_cols << "\n"
    << "Nnz: " << B_nnz << "\n";

    int const OVERHEAD = (A_nnz + B_nnz) / 4;
    auto SPGEMM_MR = rmm::mr::cuda_async_memory_resource(sizeof(ValueType) * 2 * (A_nnz + B_nnz + OVERHEAD));
    auto SPGEMM_STREAM_ALLOCATOR_INT = [&SPGEMM_MR](cudaStream_t STREAM) {return rmm::mr::thrust_allocator<int>(STREAM, SPGEMM_MR);};
    auto SPGEMM_STREAM_ALLOCATOR_LONGLONG = [&SPGEMM_MR](cudaStream_t STREAM) {return rmm::mr::thrust_allocator<long long>(STREAM, SPGEMM_MR);};
    auto SPGEMM_STREAM_ALLOCATOR_VALUETYPE = [&SPGEMM_MR](cudaStream_t STREAM) {return rmm::mr::thrust_allocator<ValueType>(STREAM, SPGEMM_MR);};
    auto SPGEMM_STREAM_ALLOCATOR_UINT8 = [&SPGEMM_MR](cudaStream_t STREAM) {return rmm::mr::thrust_allocator<uint8_t>(STREAM, SPGEMM_MR);};
    auto SPGEMM_STREAM_ALLOCATOR_UINT16 = [&SPGEMM_MR](cudaStream_t STREAM) {return rmm::mr::thrust_allocator<uint16_t>(STREAM, SPGEMM_MR);};

    auto SPGEMM_TEMPORARY_MR = rmm::mr::cuda_async_memory_resource(sizeof(char) * OVERHEAD);
    auto ASYNC_EXEC_POLICY = [&SPGEMM_TEMPORARY_MR](auto STREAM){return rmm::exec_policy_nosync(STREAM, &SPGEMM_TEMPORARY_MR);};


    // DEVICE MATRIX A --------------------
    rmm::device_vector<int>         A_d_I(A_nnz, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_A));
    rmm::device_vector<int>         A_d_J(A_nnz, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_A));
    rmm::device_vector<ValueType>   A_d_val(A_nnz, SPGEMM_STREAM_ALLOCATOR_VALUETYPE(STREAM_A));
    //-------------------------------------

    // DEVICE MATRIX B --------------------
    rmm::device_vector<int>         B_d_I(B_nnz, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B));
    rmm::device_vector<int>         B_d_J(B_nnz, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B));
    rmm::device_vector<ValueType>   B_d_val(B_nnz, SPGEMM_STREAM_ALLOCATOR_VALUETYPE(STREAM_B));
    //-------------------------------------

    thrust::copy(ASYNC_EXEC_POLICY(STREAM_A), A_I, A_I + A_nnz, A_d_I.begin());
    thrust::copy(ASYNC_EXEC_POLICY(STREAM_A), A_J, A_J + A_nnz, A_d_J.begin());
    thrust::copy(ASYNC_EXEC_POLICY(STREAM_A), A_val, A_val + A_nnz, A_d_val.begin());

    thrust::copy(ASYNC_EXEC_POLICY(STREAM_B), B_I, B_I + B_nnz, B_d_I.begin());
    thrust::copy(ASYNC_EXEC_POLICY(STREAM_B), B_J, B_J + B_nnz, B_d_J.begin());
    thrust::copy(ASYNC_EXEC_POLICY(STREAM_B), B_val, B_val + B_nnz, B_d_val.begin());

    int A_tileRows = (A_rows-1+tileSize) / tileSize;
    int A_tileCols = (A_cols-1+tileSize) / tileSize;
    int B_tileRows = (B_rows-1+tileSize) / tileSize;
    int B_tileCols = (B_cols-1+tileSize) / tileSize;
    rmm::device_vector<long long> A_participating_tiles(A_nnz, -1, SPGEMM_STREAM_ALLOCATOR_LONGLONG(STREAM_A));
    rmm::device_vector<long long> B_participating_tiles(B_nnz, -1, SPGEMM_STREAM_ALLOCATOR_LONGLONG(STREAM_B));

    dim3 A_threads_dwc{tileSize * tileSize};
    dim3 A_blocks_dwc{(A_nnz - 1 + A_threads_dwc.x)/A_threads_dwc.x};
    decide_which_tile<<<A_blocks_dwc, A_threads_dwc, 0, STREAM_A>>>
    (
        A_participating_tiles.data().get(), 
        A_d_I.data().get(), 
        A_d_J.data().get(), 
        A_nnz
    );
    dim3 B_threads_dwc{tileSize * tileSize};
    dim3 B_blocks_dwc{(B_nnz - 1 + B_threads_dwc.x)/B_threads_dwc.x};
    decide_which_tile<<<B_blocks_dwc, B_threads_dwc, 0, STREAM_B>>>
    (
        B_participating_tiles.data().get(), 
        B_d_I.data().get(), 
        B_d_J.data().get(), 
        B_nnz
    );
    
    rmm::device_vector<int> A_perTileNnz(SPGEMM_STREAM_ALLOCATOR_INT(STREAM_A));
    int cntA = 0;
    {
    thrust::sort(ASYNC_EXEC_POLICY(STREAM_A), A_participating_tiles.begin(), A_participating_tiles.end(), thrust::less<long long>());
    
    cntA = thrust::unique_count(ASYNC_EXEC_POLICY(STREAM_A), A_participating_tiles.begin(), A_participating_tiles.end());
    A_perTileNnz.resize(cntA + 1);
    thrust::reduce_by_key(ASYNC_EXEC_POLICY(STREAM_A), A_participating_tiles.begin(), A_participating_tiles.end(), thrust::make_constant_iterator<int>(1), thrust::make_discard_iterator(), A_perTileNnz.begin());
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_A),A_perTileNnz.begin(), A_perTileNnz.end(), A_perTileNnz.begin());

    auto newend = thrust::unique(ASYNC_EXEC_POLICY(STREAM_A), A_participating_tiles.begin(), A_participating_tiles.end());
    A_participating_tiles.erase(newend, A_participating_tiles.end());
    }

    rmm::device_vector<int> B_perTileNnz(SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B));
    int cntB = 0;
    {
    thrust::sort(ASYNC_EXEC_POLICY(STREAM_B), B_participating_tiles.begin(), B_participating_tiles.end(), thrust::less<long long>());
    
    cntB = thrust::unique_count(ASYNC_EXEC_POLICY(STREAM_B), B_participating_tiles.begin(), B_participating_tiles.end());
    B_perTileNnz.resize(cntB + 1);
    thrust::reduce_by_key(ASYNC_EXEC_POLICY(STREAM_B), B_participating_tiles.begin(), B_participating_tiles.end(), thrust::make_constant_iterator<int>(1), thrust::make_discard_iterator(), B_perTileNnz.begin());
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_B), B_perTileNnz.begin(), B_perTileNnz.end(), B_perTileNnz.begin());

    auto newend = thrust::unique(ASYNC_EXEC_POLICY(STREAM_B), B_participating_tiles.begin(), B_participating_tiles.end());
    B_participating_tiles.erase(newend, B_participating_tiles.end());
    }

    rmm::device_vector<int> A_d_rowPtr(A_rows + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_A));
    {
    auto zit = thrust::make_zip_iterator(thrust::make_tuple(A_d_I.begin(), A_d_J.begin(), A_d_val.begin()));
    thrust::stable_sort(ASYNC_EXEC_POLICY(STREAM_A), zit, zit + A_nnz);

    rmm::device_vector<int> A_d_rowPtr_tmp(A_rows, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_A));
    rmm::device_vector<int> A_d_index(A_rows, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_A));
    auto res = thrust::reduce_by_key(
        ASYNC_EXEC_POLICY(STREAM_A), 
        A_d_I.begin(), 
        A_d_I.end(), 
        thrust::make_constant_iterator<int>(1), 
        A_d_index.begin(), 
        A_d_rowPtr_tmp.begin());
    thrust::scatter(ASYNC_EXEC_POLICY(STREAM_A), A_d_rowPtr_tmp.begin(), res.second, A_d_index.begin(), A_d_rowPtr.begin());
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_A), A_d_rowPtr.begin(), A_d_rowPtr.end(), A_d_rowPtr.begin());
    }

    rmm::device_vector<int> B_d_rowPtr(B_rows + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B));
    {
    auto zit = thrust::make_zip_iterator(thrust::make_tuple(B_d_I.begin(), B_d_J.begin(), B_d_val.begin()));
    thrust::stable_sort(ASYNC_EXEC_POLICY(STREAM_B), zit, zit + B_nnz);

    rmm::device_vector<int> B_d_rowPtr_tmp(B_rows, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B));
    rmm::device_vector<int> B_d_index(B_rows, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B));
    auto res = thrust::reduce_by_key(
        ASYNC_EXEC_POLICY(STREAM_B), 
        B_d_I.begin(), 
        B_d_I.end(), 
        thrust::make_constant_iterator<int>(1), 
        B_d_index.begin(), 
        B_d_rowPtr_tmp.begin());
    thrust::scatter(ASYNC_EXEC_POLICY(STREAM_B), B_d_rowPtr_tmp.begin(), res.second, B_d_index.begin(), B_d_rowPtr.begin());
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_B), B_d_rowPtr.begin(), B_d_rowPtr.end(), B_d_rowPtr.begin());
    }

    dim3 A_threads_gtc {tileSize * tileSize};
    dim3 A_blocks_gtc {cntA};

    rmm::device_vector<ValueType> Atiles_vals(A_nnz, SPGEMM_STREAM_ALLOCATOR_VALUETYPE(STREAM_A));
    rmm::device_vector<uint16_t> Atiles_masks(cntA * 16, SPGEMM_STREAM_ALLOCATOR_UINT16(STREAM_A));
    rmm::device_vector<uint8_t> Atiles_rowPtr(cntA * 16, SPGEMM_STREAM_ALLOCATOR_UINT8(STREAM_A));
    rmm::device_vector<uint8_t> Atiles_rowColIdx(A_nnz, SPGEMM_STREAM_ALLOCATOR_UINT8(STREAM_A));

    cudaEventRecord(A_tileConversion_start, STREAM_A);
    generate_tiles_csr<ValueType><<<A_blocks_gtc, A_threads_gtc,0, STREAM_A>>>
    (
        Atiles_vals.data().get(),
        Atiles_masks.data().get(),
        Atiles_rowPtr.data().get(),
        Atiles_rowColIdx.data().get(),
        A_participating_tiles.data().get(), 
        cntA, 
        A_perTileNnz.data().get(), 
        A_d_J.data().get(), 
        A_d_val.data().get(), 
        A_d_rowPtr.data().get(),
        A_rows
    );
    cudaEventRecord(A_tileConversion_end, STREAM_A);

    dim3 B_threads_gtc {tileSize * tileSize};
    dim3 B_blocks_gtc {cntB};

    rmm::device_vector<ValueType> Btiles_vals(B_nnz, SPGEMM_STREAM_ALLOCATOR_VALUETYPE(STREAM_B));
    rmm::device_vector<uint16_t> Btiles_masks(cntB * 16, SPGEMM_STREAM_ALLOCATOR_UINT16(STREAM_B));
    rmm::device_vector<uint8_t> Btiles_rowPtr(cntB * 16, SPGEMM_STREAM_ALLOCATOR_UINT8(STREAM_B));
    rmm::device_vector<uint8_t> Btiles_rowColIdx(B_nnz, SPGEMM_STREAM_ALLOCATOR_UINT8(STREAM_B));

    cudaEventRecord(B_tileConversion_start, STREAM_B);
    generate_tiles_csr<ValueType><<<B_blocks_gtc, B_threads_gtc,0, STREAM_B>>>
    (
        Btiles_vals.data().get(),
        Btiles_masks.data().get(),
        Btiles_rowPtr.data().get(),
        Btiles_rowColIdx.data().get(),
        B_participating_tiles.data().get(), 
        cntB, 
        B_perTileNnz.data().get(), 
        B_d_J.data().get(), 
        B_d_val.data().get(), 
        B_d_rowPtr.data().get(),
        B_rows
    );
    cudaEventRecord(B_tileConversion_end, STREAM_B);

    rmm::device_vector<uint16_t> Btiles_transposed_mask(cntB * 16, SPGEMM_STREAM_ALLOCATOR_UINT16(STREAM_B));
    dim3 threads_tBm {256};
    dim3 blocks_tBm {(cntB-1+threads_tBm.x)/threads_tBm.x};
    __transpose_B_mask<<<blocks_tBm, threads_tBm, 0, STREAM_B>>>(Btiles_transposed_mask.data().get(), Btiles_masks.data().get(), cntB);

    // create High level Representation of A -> A_
    rmm::device_vector<int> _A_tileRowPtr(A_tileRows + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_D));
    rmm::device_vector<int> _A_tileColIdx(cntA, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_D));
    {
    rmm::device_vector<int> _A_tileRowPtr_tmp(A_tileRows, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_D));
    auto newendA = thrust::reduce_by_key(
        ASYNC_EXEC_POLICY(STREAM_D),
        thrust::make_transform_iterator(A_participating_tiles.begin(), getHigh32()),
        thrust::make_transform_iterator(A_participating_tiles.end(), getHigh32()),
        thrust::make_constant_iterator<int>(1),
        _A_tileColIdx.begin(), // borrow _A_tileColIdx as temporary index buffer
        _A_tileRowPtr_tmp.begin()
    );
    thrust::scatter(ASYNC_EXEC_POLICY(STREAM_D), _A_tileRowPtr_tmp.begin(), newendA.second, _A_tileColIdx.begin(), _A_tileRowPtr.begin());
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_D), _A_tileRowPtr.begin(), _A_tileRowPtr.end(), _A_tileRowPtr.begin());

    thrust::copy(
        ASYNC_EXEC_POLICY(STREAM_D),
        thrust::make_transform_iterator(A_participating_tiles.begin(), getLow32()),
        thrust::make_transform_iterator(A_participating_tiles.end(), getLow32()),
        _A_tileColIdx.begin()
    );
    }

    // create High level Representation of B -> B_
    rmm::device_vector<int> _B_tileRowPtr(B_tileRows + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    rmm::device_vector<int> _B_tileColIdx(cntB, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    {
    rmm::device_vector<int> _B_tileRowPtr_tmp(B_tileRows, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    auto newendB = thrust::reduce_by_key(
        ASYNC_EXEC_POLICY(STREAM_E),
        thrust::make_transform_iterator(B_participating_tiles.begin(), getHigh32()),
        thrust::make_transform_iterator(B_participating_tiles.end(), getHigh32()),
        thrust::make_constant_iterator<int>(1),
        _B_tileColIdx.begin(), // borrow _B_tileColIdx as temporary index buffer
        _B_tileRowPtr_tmp.begin()
    );
    thrust::scatter(ASYNC_EXEC_POLICY(STREAM_E), _B_tileRowPtr_tmp.begin(), newendB.second, _B_tileColIdx.begin(), _B_tileRowPtr.begin());
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_E), _B_tileRowPtr.begin(), _B_tileRowPtr.end(), _B_tileRowPtr.begin());

    thrust::copy(
        ASYNC_EXEC_POLICY(STREAM_E),
        thrust::make_transform_iterator(B_participating_tiles.begin(), getLow32()),
        thrust::make_transform_iterator(B_participating_tiles.end(), getLow32()),
        _B_tileColIdx.begin()
    );
    }
    
    cudaStreamSynchronize(STREAM_B);
    rmm::device_vector<int> _B_tileOffsets(cntB, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    thrust::sequence(ASYNC_EXEC_POLICY(STREAM_E), _B_tileOffsets.begin(), _B_tileOffsets.end());
    thrust::transform(ASYNC_EXEC_POLICY(STREAM_E), B_participating_tiles.begin(), B_participating_tiles.end(), B_participating_tiles.begin(), swap32());
    {
    auto zit = thrust::make_zip_iterator(thrust::make_tuple(B_participating_tiles.begin(), _B_tileOffsets.begin()));
    thrust::sort(ASYNC_EXEC_POLICY(STREAM_E), zit, zit+B_participating_tiles.size());
    }

    rmm::device_vector<int> _B_tileColPtr(B_tileCols + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    rmm::device_vector<int> _B_tileRowIdx(cntB, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    {
    rmm::device_vector<int> _B_tileColPtr_tmp(B_tileCols, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    auto newendBB = thrust::reduce_by_key(
        ASYNC_EXEC_POLICY(STREAM_E),
        thrust::make_transform_iterator(B_participating_tiles.begin(), getHigh32()),
        thrust::make_transform_iterator(B_participating_tiles.end(), getHigh32()),
        thrust::make_constant_iterator<int>(1),
        _B_tileRowIdx.begin(), // borrow _B_tileRowIdx as temporary index buffer
        _B_tileColPtr_tmp.begin()
    );
    thrust::scatter(ASYNC_EXEC_POLICY(STREAM_E), _B_tileColPtr_tmp.begin(), newendBB.second, _B_tileRowIdx.begin(), _B_tileColPtr.begin());
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_E), _B_tileColPtr.begin(), _B_tileColPtr.end(), _B_tileColPtr.begin());
    thrust::copy(
        ASYNC_EXEC_POLICY(STREAM_E),
        thrust::make_transform_iterator(B_participating_tiles.begin(), getLow32()),
        thrust::make_transform_iterator(B_participating_tiles.end(), getLow32()),
        _B_tileRowIdx.begin()
    );
    }

    cudaDeviceSynchronize();
    auto tileCSR_conversion_end = std::chrono::high_resolution_clock::now();
    float tileCSR_conversion_time = std::chrono::duration<float, std::milli>(tileCSR_conversion_end-tileCSR_conversion_start).count();

    std::jthread count_flop([=] (unsigned long long *flop)
    {
        cudaStreamSynchronize(STREAM_B);
        thrust::host_vector<int> B_rowPtr(B_rows+1);
        thrust::copy(B_d_rowPtr.begin(), B_d_rowPtr.end(), B_rowPtr.begin());

        for(int i = 0; i < A_nnz; ++i)
        {
            int rowidx = A_J[i];
            *flop += B_rowPtr[rowidx+1] - B_rowPtr[rowidx];
        }
    }, &flop);

    rmm::device_vector<int>().swap(A_d_I);
    rmm::device_vector<int>().swap(A_d_J);
    rmm::device_vector<ValueType>().swap(A_d_val);

    rmm::device_vector<int>().swap(B_d_I);
    rmm::device_vector<int>().swap(B_d_J);
    rmm::device_vector<ValueType>().swap(B_d_val);

    rmm::device_vector<long long>().swap(A_participating_tiles);
    rmm::device_vector<long long>().swap(B_participating_tiles);

    rmm::device_vector<int>().swap(A_d_rowPtr);
    rmm::device_vector<int>().swap(B_d_rowPtr);



    //----TIMERS------
    float Aconversion, Bconversion;
    float hlm_time[REPEAT], sp0_time[REPEAT], sp1_time[REPEAT], aC_time[REPEAT], sO_time[REPEAT], accumulator_time[REPEAT];
    float allocate_c_time[REPEAT], pem_spgemm_time[REPEAT], kernel_time[REPEAT], malloc_time[REPEAT];
    //----------------

    //----BUFFERS-----
    int _C_nnz = 0; // nonzero of C'
    int C_nnz = 0; // nonzero of C

    int * _C_rowPtr;
    int *_C_tileRowIdx, *_C_tileColIdx; 
    int *pairs_insertion_offset;
    int *d_pairs_a, *d_pairs_b;
    unsigned *Ctiles_mask;
    int *_C_perTileNnz;
    ValueType *Ctiles_vals;
    uint8_t *Ctiles_rowColIdx;
    uint8_t *Ctiles_rowPtr;
    //----------------
    
    auto loop_cleanup = [&]
    {
        cudaFree(Ctiles_vals);
        cudaFree(Ctiles_rowColIdx);
        cudaFree(_C_perTileNnz);
        cudaFree(Ctiles_mask);
        cudaFree(d_pairs_b);
        cudaFree(d_pairs_a);
        cudaFree(pairs_insertion_offset);
        cudaFree(_C_tileColIdx);
        cudaFree(_C_tileRowIdx);
        cudaFree(_C_rowPtr);
        cudaFree(Ctiles_rowPtr);
    };

    for(int n = 0; n < REPEAT; ++n)
    {
        //<<<-------------------------ALGORITHM------------------------->>>
        auto pem_spgemm_start = std::chrono::high_resolution_clock::now();
        
        cudaMallocAsync(&_C_rowPtr, sizeof(int) * (A_tileRows+1), STREAM_C);
        sfBIN bin;

        cudaDeviceSynchronize();
        auto step1_start = std::chrono::high_resolution_clock::now();
        if(B_tileCols > 512 * 32)
        {
            if(n == 0) printf("\nstep1 using NSPARSE\n");
            init_bin(&bin, A_tileRows);
            set_max_bin(_A_tileRowPtr.data().get(), _A_tileColIdx.data().get(), _B_tileRowPtr.data().get(), &bin, A_tileRows);
            set_row_nnz(_A_tileRowPtr.data().get(), _A_tileColIdx.data().get(), _B_tileRowPtr.data().get(), _B_tileColIdx.data().get(),
            _C_rowPtr, &bin, A_tileRows, &_C_nnz);
            set_min_bin(&bin, A_tileRows);
        }
        else
        {
            if(n == 0) printf("\nstep1 using TileSpGEMM's\n");
            int step1_numthreads = 128;
            int step1_numblocks = ceil((double)A_tileRows / (double)4);
            cudaEventRecord(high_level_multiply_start, STREAM_C);
            tile_spgemm_step1_cuda_spa_kernel<<<step1_numblocks, step1_numthreads, 0, STREAM_C>>>
            (
                _A_tileRowPtr.data().get(),
                _A_tileColIdx.data().get(),
                A_tileRows,
                _B_tileRowPtr.data().get(),
                _B_tileColIdx.data().get(),
                B_tileCols,
                _C_rowPtr
            );
            thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_C), _C_rowPtr, _C_rowPtr + (A_tileRows+1), _C_rowPtr);
            cudaMemcpyAsync(&_C_nnz, &_C_rowPtr[A_tileRows], sizeof(int), cudaMemcpyDeviceToHost, STREAM_C);
        }

        cudaMallocAsync(&_C_tileColIdx, _C_nnz * sizeof(int), STREAM_C);
        cudaMallocAsync(&_C_tileRowIdx, _C_nnz * sizeof(int), STREAM_C);

        if(B_tileCols > 512 * 32)
        {
            cudaDeviceSynchronize();
            calculate_value_col_bin
            (
                _A_tileRowPtr.data().get(),
                _A_tileColIdx.data().get(),
                nullptr,
                _B_tileRowPtr.data().get(),
                _B_tileColIdx.data().get(),
                nullptr,
                _C_rowPtr,
                _C_tileRowIdx,
                _C_tileColIdx,
                nullptr,
                &bin,
                A_tileRows,
                B_tileCols
            );
            release_bin(bin);
        }
        else
        {
            int step1_numthreads = 128;
            int step1_numblocks = ceil((double)A_tileRows / (double)4);
            tile_spgemm_step1_numeric_cuda_spa_kernel<<<step1_numblocks, step1_numthreads, 0, STREAM_C>>>
            (
                _A_tileRowPtr.data().get(),
                _A_tileColIdx.data().get(),
                A_tileRows,
                _B_tileRowPtr.data().get(),
                _B_tileColIdx.data().get(),
                B_tileCols,
                _C_rowPtr,
                _C_tileRowIdx,
                _C_tileColIdx,
                nullptr,
                nullptr,
                nullptr
            );
            cudaEventRecord(high_level_multiply_end, STREAM_C);
        }
        cudaDeviceSynchronize();
        auto step1_end = std::chrono::high_resolution_clock::now();

        if(n == 0) std::cout << "\nstep2 pemSpGEMM\n";
        dim3 threads_sp{256};
        dim3 blocks_sp{(_C_nnz-1+threads_sp.x)/threads_sp.x};

        cudaMallocAsync(&pairs_insertion_offset, sizeof(int) * (_C_nnz+1), STREAM_C);

        cudaEventRecord(allocate_c_start, STREAM_C);
        pem_spgemm_step2_search_pairs<0><<<blocks_sp, threads_sp, 0, STREAM_C>>>
        (
            nullptr,
            nullptr,
            _C_tileColIdx,
            _A_tileRowPtr.data().get(),
            _A_tileColIdx.data().get(),
            _B_tileColPtr.data().get(),
            _B_tileRowIdx.data().get(),
            A_tileRows,
            _C_nnz,
            _B_tileOffsets.data().get(),
            _C_tileRowIdx,
            pairs_insertion_offset
        );
        thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_C), pairs_insertion_offset, pairs_insertion_offset + (_C_nnz+1), pairs_insertion_offset);
        cudaEventRecord(sp0_end, STREAM_C);

        int d_pairs_count = 0;
        cudaMemcpyAsync(&d_pairs_count, &pairs_insertion_offset[_C_nnz], sizeof(int), cudaMemcpyDeviceToHost, STREAM_C);

        cudaMallocAsync(&d_pairs_a, sizeof(int) * d_pairs_count, STREAM_C); 
        cudaMallocAsync(&d_pairs_b, sizeof(int) * d_pairs_count, STREAM_C); 

        cudaEventRecord(sp1_start, STREAM_C);
        pem_spgemm_step2_search_pairs<1><<<blocks_sp, threads_sp, 0, STREAM_C>>>
        (
            d_pairs_a,
            d_pairs_b,
            _C_tileColIdx,
            _A_tileRowPtr.data().get(),
            _A_tileColIdx.data().get(),
            _B_tileColPtr.data().get(),
            _B_tileRowIdx.data().get(),
            A_tileRows,
            _C_nnz,
            _B_tileOffsets.data().get(),
            _C_tileRowIdx,
            pairs_insertion_offset
        );
        cudaEventRecord(sp1_end, STREAM_C);

        cudaMallocAsync(&Ctiles_mask, sizeof(unsigned) * 8 * _C_nnz, STREAM_C);
        cudaMallocAsync(&_C_perTileNnz, sizeof(int) * (_C_nnz+1), STREAM_C);

        dim3 threads_aC {128};
        dim3 blocks_aC {(_C_nnz+15)/16};

        cudaEventRecord(aC_start, STREAM_C);
        pem_spgemm_step2_compute_CMasksAndOffsets<<<blocks_aC, threads_aC, 0, STREAM_C>>>
        (
            d_pairs_a,
            d_pairs_b,
            _C_nnz,
            _C_perTileNnz,
            Atiles_masks.data().get(),
            Btiles_masks.data().get(),
            Btiles_transposed_mask.data().get(),
            pairs_insertion_offset,
            Ctiles_mask
        );
        thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_C), _C_perTileNnz, _C_perTileNnz + (_C_nnz+1), _C_perTileNnz);
        cudaEventRecord(aC_end, STREAM_C);

        cudaMemcpyAsync(&C_nnz, &_C_perTileNnz[_C_nnz], sizeof(int), cudaMemcpyDeviceToHost, STREAM_C);

        cudaMallocAsync(&Ctiles_vals, sizeof(ValueType) * C_nnz, STREAM_C);
        cudaMallocAsync(&Ctiles_rowColIdx, sizeof(uint8_t) * C_nnz, STREAM_C);
        cudaMallocAsync(&Ctiles_rowPtr, sizeof(uint8_t) * 16 * _C_nnz, STREAM_C);

        dim3 threads_Cs{128};
        dim3 blocks_Cs{(_C_nnz+7)/8};

        cudaEventRecord(setOffset_start, STREAM_C);
        pem_spgemm_step2_compute_CrowColIdx<<<blocks_Cs, threads_Cs, 0, STREAM_C>>>
        (
            _C_nnz,
            _C_perTileNnz,
            Ctiles_rowColIdx,
            Ctiles_rowPtr,
            Ctiles_mask
        );
        cudaEventRecord(allocate_c_end, STREAM_C);

        if(n == 0) std::cout << "\nstep3 pemSpGEMM\n\n\n";

        dim3 threads_mp {128};
        dim3 blocks_mp {(_C_nnz+3)/4};

        pem_spgemm_step3_accumulate<ValueType><<<blocks_mp, threads_mp, 0, STREAM_C>>>
        (
            d_pairs_a,
            d_pairs_b,
            _C_nnz,
            Ctiles_vals,
            _C_perTileNnz,
            Ctiles_rowColIdx,
            A_perTileNnz.data().get(),
            Atiles_vals.data().get(),
            Atiles_masks.data().get(),
            Atiles_rowPtr.data().get(),
            B_perTileNnz.data().get(),
            Btiles_vals.data().get(),
            Btiles_masks.data().get(),
            Btiles_rowPtr.data().get(),
            Btiles_transposed_mask.data().get(),
            pairs_insertion_offset
        );
        cudaEventRecord(accumulator_end, STREAM_C);
        cudaDeviceSynchronize();
        //<<<---------------------------------------------->>>


        auto pem_spgemm_end = std::chrono::high_resolution_clock::now();
        auto pem_spgemm_duration = std::chrono::duration<float, std::milli>(pem_spgemm_end-pem_spgemm_start);

        hlm_time[n] = std::chrono::duration<float, std::milli>(step1_end-step1_start).count();
        cudaEventElapsedTime(&sp0_time[n], allocate_c_start, sp0_end);
        cudaEventElapsedTime(&sp1_time[n], sp1_start, sp1_end);
        cudaEventElapsedTime(&aC_time[n], aC_start, aC_end);
        cudaEventElapsedTime(&sO_time[n], setOffset_start, allocate_c_end);
        cudaEventElapsedTime(&accumulator_time[n], allocate_c_end, accumulator_end);

        allocate_c_time[n] = sp0_time[n] + sp1_time[n] + aC_time[n] + sO_time[n];

        pem_spgemm_time[n] = pem_spgemm_duration.count();
        kernel_time[n] = hlm_time[n] + allocate_c_time[n] + accumulator_time[n];
        malloc_time[n] = pem_spgemm_time[n] - kernel_time[n];

        if(n < REPEAT-1) loop_cleanup();
    }

    auto fastest = std::min_element(pem_spgemm_time, pem_spgemm_time + REPEAT);
    int fidx = std::distance(pem_spgemm_time, fastest); 
    std::cout << "fidx: " << fidx << "\n\n";

    cudaEventElapsedTime(&Aconversion, A_tileConversion_start, A_tileConversion_end);
    cudaEventElapsedTime(&Bconversion, B_tileConversion_start, B_tileConversion_end);
    
    count_flop.join();
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "<---Program done--->\n";
    std::cout << "Matrix A CSR to tile kernel took---------" << Aconversion << "ms\n";
    std::cout << "Matrix B CSR to tile kernel took---------" << Bconversion << "ms\n";
    std::cout << "total------------------------------------" << tileCSR_conversion_time << "ms\n\n";

    std::cout << "step1 - High Level Multiplication took---" << hlm_time[fidx] << "ms\n";
    std::cout << "step2 - Allocating C took----------------" << allocate_c_time[fidx] << "ms\n";
    std::cout << "step3 - Accumulation took----------------" << accumulator_time[fidx] << "ms\n\n";
    
    double GFlops = flop*2.0/(pem_spgemm_time[fidx] * 1e6);
    double compression_ratio = static_cast<double>(flop) / C_nnz;

    std::cout << "pemSpGEMM took " 
    << pem_spgemm_time[fidx] << "ms ----- GFlops: " << GFlops 
    << "\nKernel time " << kernel_time[fidx] << "ms\nmalloc time " << malloc_time[fidx] << "ms\n";
    std::cout << "Flop count: " << flop << "\n\n";
    std::cout << "C tiles: " << _C_nnz << "\n";
    std::cout << "C nnz: " << C_nnz << "\n";
    std::cout << "Compression ratio " << compression_ratio << "\n";

    char const *csv_filepath = "./pemspgemm_benchmark_result.csv";
    std::ofstream write_csv;
    write_csv.open(csv_filepath, std::ios::app);

    std::regex get_name(".*\/(.*)\.mtx");
    std::string const input_path = argv[1];
    std::smatch input_name;
    std::regex_search(input_path.begin(), input_path.end(), input_name, get_name);
    write_csv << std::fixed << std::setprecision(2);
    write_csv 
    << "\n"
    << input_name[1] << "," 
    << flop << "," 
    << C_nnz << ","
    << compression_ratio << ","
    << hlm_time[fidx] << "," 
    << allocate_c_time[fidx] << "," 
    << accumulator_time[fidx] << "," 
    << pem_spgemm_time[fidx] << "," 
    << GFlops << "," 
    << Aconversion << ","
    << tileCSR_conversion_time;

    write_csv.close();

    auto cleanup = [&]
    {
        std::cout << "CLEANING UP RESOURCES\n\n";
        cudaEventDestroy(A_tileConversion_start);
        cudaEventDestroy(A_tileConversion_end);
        cudaEventDestroy(B_tileConversion_start);
        cudaEventDestroy(B_tileConversion_end);
        cudaEventDestroy(high_level_multiply_start);
        cudaEventDestroy(high_level_multiply_end);
        cudaEventDestroy(cusparse_start);
        cudaEventDestroy(cusparse_end);
        cudaEventDestroy(allocate_c_start);
        cudaEventDestroy(sp0_end);
        cudaEventDestroy(sp1_start);
        cudaEventDestroy(sp1_end);
        cudaEventDestroy(aC_start);
        cudaEventDestroy(aC_end);
        cudaEventDestroy(setOffset_start);
        cudaEventDestroy(allocate_c_end);
        cudaEventDestroy(accumulator_end);
        cudaEventDestroy(sanitize_C_start);
        cudaEventDestroy(sanitize_C_end);

        cudaFree(_C_tileRowIdx);
        cudaFree(_C_tileColIdx);
        cudaFree(d_pairs_a);
        cudaFree(d_pairs_b);
        cudaFree(Ctiles_mask);
        cudaFree(Ctiles_rowColIdx);
        cudaFree(Ctiles_vals);
        cudaFree(Ctiles_rowPtr);
    };

    if(!atoi(argv[2]))
    {
        std::cout << "Not saving results. Exiting.\n";
        cleanup();
        std::atexit([]{cudaDeviceReset();});
        return 0;
    }

    cudaFreeAsync(pairs_insertion_offset, STREAM_C);

    rmm::device_vector<int> Crows(C_nnz, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_C));
    rmm::device_vector<int> Ccols(C_nnz, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_C));

    dim3 threads_sC {tileSize * tileSize};
    dim3 blocks_sC {(_C_nnz+7)/8};

    cudaEventRecord(sanitize_C_start, STREAM_C);
    sanitize_C<<<blocks_sC, threads_sC, 0, STREAM_C>>>
    (
        Crows.data().get(), 
        Ccols.data().get(), 
        Ctiles_rowColIdx,
        Ctiles_vals,
        _C_nnz,
        _C_tileRowIdx,
        _C_tileColIdx, 
        _C_perTileNnz
    );
    cudaEventRecord(sanitize_C_end, STREAM_C);

    //temporary
    {
        auto zit = thrust::make_zip_iterator(Crows.begin(), Ccols.begin(), Ctiles_vals);
        thrust::stable_sort(ASYNC_EXEC_POLICY(STREAM_C), zit, zit + C_nnz);
    }

    cudaStreamSynchronize(STREAM_C);

    float sanitize_C_time;
    cudaEventElapsedTime(&sanitize_C_time, sanitize_C_start, sanitize_C_end);
    std::cout << "sanitize_C took " << sanitize_C_time << "ms\n";

    std::cout << "Saving results to /tmp/SPGEMM_RESULT_*.txt\n";
    char const *filename0 = "/tmp/SPGEMM_RESULT_NNZ.txt";
    char const *filename1 = "/tmp/SPGEMM_RESULT_ROWS.txt";
    char const *filename2 = "/tmp/SPGEMM_RESULT_COLS.txt";
    char const *filename3 = "/tmp/SPGEMM_RESULT_VALS.txt";

    std::ofstream outfile;

    auto print = [&outfile] (auto v) { outfile << v << "\n"; };
    
    thrust::host_vector<int> hCrows(Crows.size());
    thrust::host_vector<int> hCcols(Ccols.size());
    std::vector<ValueType> hCvals(C_nnz);

    thrust::copy(Crows.begin(), Crows.end(), hCrows.begin());
    thrust::copy(Ccols.begin(), Ccols.end(), hCcols.begin());
    cudaMemcpy(hCvals.data(), Ctiles_vals, sizeof(ValueType) * C_nnz, cudaMemcpyDeviceToHost);

    outfile.open(filename0, std::ios::out);
    outfile << C_nnz;
    outfile.close();

    outfile.open(filename1, std::ios::out);
    std::for_each(hCrows.begin(), hCrows.end(), print);
    outfile.close();

    outfile.open(filename2, std::ios::out);
    std::for_each(hCcols.begin(), hCcols.end(), print);
    outfile.close();

    outfile.open(filename3, std::ios::out);
    outfile << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::for_each(hCvals.begin(), hCvals.end(), print);
    outfile.close();


    cleanup();
    std::atexit([]{cudaDeviceReset();});
    return 0; // <--------------------------------------------------------------------------------------------------------------------------------

    // streams are destroyed by rmm
}
