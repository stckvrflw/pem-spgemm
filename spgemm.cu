/*
Author: Petrus E. Manurung
*/

#include <cstdlib>
#include <cstdio>

#include <fstream>
#include <thread>
#include <algorithm>
#include <chrono>

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>

#include <cuda/pipeline>
#include <mma.h>

#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/unique.h>
#include <thrust/partition.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>

#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <cusparse.h>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/device_vector.hpp>

#include "cx.h"
#include "fast_matrix_market/fast_matrix_market.hpp"
#include "TileCSR.h"
#include "utilities.h"

namespace cg = cooperative_groups;

template<typename ValueType>
void read_matrix_market
(
    char const *matrix_path,
    thrustHvecPin<int> &I, 
    thrustHvecPin<int> &J, 
    thrustHvecPin<ValueType> &vals, 
    int &rows, int &cols, int &nnz
)
{
    std::ifstream file(matrix_path);
    std::vector<int> _I;
    std::vector<int> _J;
    std::vector<ValueType> _vals;
    int _rows, _cols, _nnz;
    fast_matrix_market::read_matrix_market_triplet(file,
                                                    _rows, _cols,
                                                    _I, _J, _vals);
    _nnz = _vals.size();
    
    I.reserve(_nnz);
    J.reserve(_nnz);
    vals.reserve(_nnz);

    rows = _rows;
    cols = _cols;
    nnz = _nnz;

    I = _I;
    J = _J;
    vals = _vals;
    file.close();
}

template<unsigned tileSize = 16>
__global__ void __launch_bounds__(tileSize * tileSize)
decide_which_tile
(
    r_Ptr<long long> participating_tiles,
    cr_Ptr<int> d_I,
    cr_Ptr<int> d_J,
    int nnz
)
{
    auto grid = cg::this_grid();
    auto my_tid = grid.thread_rank();
    // out of range check
    if(my_tid >= nnz) return;
    int my_y = d_I[my_tid];
    int my_x = d_J[my_tid];
    // int this_tile_y = my_y / tileSize;
    // int this_tile_x = my_x / tileSize;
    int this_tile_y = my_y >> 4;
    int this_tile_x = my_x >> 4;

    // long long my_tile = (static_cast<long long>(this_tile_y) << 32) | this_tile_x;
    long long my_tile = this_tile_y;
    my_tile <<= 32;
    my_tile |= this_tile_x;
    participating_tiles[my_tid] = my_tile;
}

template<typename ValueType, int tileSize = 16>
__global__ void __launch_bounds__(tileSize * tileSize)
generate_tiles_csr
(
    r_Ptr<TileCSR_rev<ValueType, tileSize>> d_tiles,
    r_Ptr<ValueType> d_tiles_vals,
    r_Ptr<uint8_t> d_tiles_rowColIdx,
    cr_Ptr<long long> participating_tiles,
    cr_Ptr<int> participating_tiles_size,
    r_Ptr<int> perTileNnz,
    cr_Ptr<int> d_J,
    cr_Ptr<ValueType> d_vals,
    cr_Ptr<int> d_rowPtr,
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
    while(block_id < *participating_tiles_size) 
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

        auto my_row_group = cg::tiled_partition<16>(block); // swap 16 to tileSize
        
        int my_row_group_rowPtr_offset = block_tile_offset_y + my_row_group.meta_group_rank();
        // if(my_row_group_rowPtr_offset >= d_rowPtr_size) return;
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
        unsigned my_row_nnz = __popc(my_row_group.ballot(thread_J != -1));
        MaskType my_row_mask = thread_J != -1 ? 1 : 0;

        my_row_mask <<= (thread_J%tileSize);
        // my_row_mask |=  my_row_group.shfl_down(my_row_mask, 8);
        // my_row_mask |=  my_row_group.shfl_down(my_row_mask, 4);
        // my_row_mask |=  my_row_group.shfl_down(my_row_mask, 2);
        // my_row_mask |=  my_row_group.shfl_down(my_row_mask, 1);
        my_row_mask = cg::reduce(my_row_group, my_row_mask, cg::bit_or<decltype(my_row_mask)>());

        if(my_row_group.thread_rank() == 0) {
            d_tiles[block_d_tiles_offset].mask[my_row_group.meta_group_rank()] = my_row_mask;
            temp_buffer[my_row_group.meta_group_rank()] = my_row_nnz;
        }
        block.sync();

        if(my_row_group.meta_group_rank() == 0) {
            my_row_nnz = temp_buffer[my_row_group.thread_rank()];
            my_row_nnz = cg::exclusive_scan(my_row_group, my_row_nnz);
            d_tiles[block_d_tiles_offset].rowPtr[my_row_group.thread_rank()] = my_row_nnz; 
        }

        int tile_offset = perTileNnz[block_id];

        if(block.thread_rank() == 0) {
            d_tiles[block_d_tiles_offset].vals = d_tiles_vals + tile_offset;
            d_tiles[block_d_tiles_offset].rowColIdx = d_tiles_rowColIdx + tile_offset;
        }
        block.sync();

        int my_loc = (thread_val != 0) ? 1 : 0;
        using BlockScan = cub::BlockScan<int, 256>;
        __shared__ typename BlockScan::TempStorage temp_storage;
        BlockScan(temp_storage).ExclusiveSum(my_loc, my_loc);

        if(thread_J != -1)
        {
            d_tiles[block_d_tiles_offset].vals[my_loc] = thread_val;
            d_tiles[block_d_tiles_offset].rowColIdx[my_loc] = my_RowColIdx;
        }

        block_id += grid.num_blocks();
    }
}

template<typename ValueType>
__global__ void __launch_bounds__(256)
__transpose_B_mask
(
    r_Ptr<uint16_t> Btiles_transposed_mask,
    cr_Ptr<TileCSR_rev<ValueType>> Btiles, 
    int Btiles_size
)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint16_t thread_buf[16];
    while(idx < Btiles_size)
    {
        uint16_t tile_mask[16];
        *(reinterpret_cast<ulonglong4*>(tile_mask)) = *(reinterpret_cast<ulonglong4 const*>(Btiles[idx].mask));
        
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
        *(reinterpret_cast<ulonglong4*>(&Btiles_transposed_mask[(idx*16)])) = *(reinterpret_cast<ulonglong4*>(thread_buf));

        idx += (gridDim.x << 8);
    }
}

__attribute__((optimize("O3")))
int cusparse_highLevelMultiply
(
    r_Ptr<int> dA_csrOffsets,
    r_Ptr<int> dA_columns,
    r_Ptr<half> dA_values,
    int A_num_rows,
    int A_num_cols,
    int A_nnz,
    r_Ptr<int> dB_csrOffsets,
    r_Ptr<int> dB_columns,
    r_Ptr<half> dB_values,
    int B_num_rows,
    int B_num_cols,
    int B_nnz,
    int **d_CtileRowIdx,
    int **d_CtileColIdx,
    cudaStream_t stream,
    rmm::exec_policy_nosync ASYNC_EXEC_POLICY,
    cudaEvent_t &cusparse_start,
    cudaEvent_t &cusparse_end,
    cudaEvent_t &start,
    cudaEvent_t &end,
    std::chrono::high_resolution_clock::time_point &pem_spgemm_start
)
{
    int *dC_csrOffsets;
    int *dC_columns;
    int *dC_rows;
    half *dC_values;

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;

    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_16F;
    half               alpha       = 1.0;
    half               beta        = 0.0;

    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;

    CHECK_CUSPARSE( cusparseCreate(&handle) )
    CHECK_CUSPARSE( cusparseSetStream(handle, stream) )
    cudaEventRecord(cusparse_start, stream);
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                      dB_csrOffsets, dB_columns, dB_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                      dC_csrOffsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )

    // allocate C offsets
    CHECK_CUDA( cudaMallocAsync((void**) &dC_csrOffsets,
                           (A_num_rows + 1) * sizeof(int), stream) )

    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )
    
    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL) )
    CHECK_CUDA( cudaMallocAsync((void**) &dBuffer1, bufferSize1, stream) )
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1) )

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL) )
    CHECK_CUDA( cudaMallocAsync((void**) &dBuffer2, bufferSize2, stream) )
    
    pem_spgemm_start = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start, stream);
    // compute the intermediate product of A * B
    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, matA, matB, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2) )
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                         &C_nnz1) )
    // allocate matrix C
    CHECK_CUDA( cudaMallocAsync((void**) &dC_columns, C_nnz1 * sizeof(int), stream)   )
    CHECK_CUDA( cudaMallocAsync((void**) &dC_values,  C_nnz1 * sizeof(half), stream) )
    CHECK_CUDA( cudaMallocAsync((void**) &dC_rows, C_nnz1 * sizeof(int), stream) )

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values) )

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

    CHECK_CUSPARSE( cusparseXcsr2coo(handle, dC_csrOffsets, C_nnz1, C_num_rows1, dC_rows, CUSPARSE_INDEX_BASE_ZERO) )
    cudaEventRecord(end, stream);

    // this is not wrong, swapped because Xcsr2coo is confusing
    *d_CtileColIdx = dC_rows;
    *d_CtileRowIdx = dC_columns;

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    cudaFreeAsync(dC_csrOffsets, stream);
    cudaFreeAsync(dC_values, stream);

    cudaEventRecord(cusparse_end, stream);

    return C_nnz1;
}

template<int pass>
__device__ __forceinline__
// we iterate on lane_iter, search on lane_targ
int __find_pairs
(
    auto warp,
    r_Ptr<int> pairs_a,
    r_Ptr<int> pairs_b,
    r_Ptr<int> C_targetTile,
    int w_start,
    cr_Ptr<int> lane_iter,
    int lane_iter_len,
    cr_Ptr<int> lane_targ,
    int lane_targ_len,
    int iter_offset,
    int targ_offset,
    int AorB,
    cr_Ptr<int> B_tileOffsets,
    int insertion_start
)
{
    int __local_count = 0; // for use when pass = 0 only
    for(int i = warp.thread_rank(); i < lane_iter_len; i+=32) {
        auto loop_participants = cg::coalesced_threads();
        int found = binarySearch(lane_targ, lane_iter[i], lane_targ_len);
        
        if constexpr(pass == 1)
        {
        loop_participants.sync();
        }
        if(found != -1) {
            if constexpr(pass == 1) // 1 = second pass
            {   
            auto lucky_ones = cg::coalesced_threads();
            int first = iter_offset + i;
            int second = targ_offset + found;
            if(!AorB) // meaning its B, need to swap-- order is AB
            std::swap(first, second);
            second = B_tileOffsets[second];
            int targ_idx = insertion_start + lucky_ones.thread_rank();
            pairs_a[targ_idx] = first;
            pairs_b[targ_idx] = second;
            C_targetTile[targ_idx] = w_start;
            insertion_start+=lucky_ones.num_threads();
            }
            else // 0 = first pass, only count how many are there
            ++__local_count;
        }
        if constexpr(pass == 1)
        {
        insertion_start = cg::reduce(loop_participants, insertion_start, cg::greater<int>());
        }
    }
    return __local_count;
}

template<int pass>
__global__ void __launch_bounds__(256) 
search_pairs
(
    r_Ptr<int> pairs_a,
    r_Ptr<int> pairs_b,
    r_Ptr<int> C_targetTile,
    cr_Ptr<int> C_colIdx,
    cr_Ptr<int> A_rowPtr,
    cr_Ptr<int> A_colIdx,
    cr_Ptr<int> B_colPtr,
    cr_Ptr<int> B_rowIdx,
    int C_rowPtr_size,
    int C_colIdx_size,
    cr_Ptr<int> B_tileOffsets,
    cr_Ptr<int> C_rowIdx,
    r_Ptr<int> pairs_insertion_offset
)
{
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int w_start = (blockIdx.x << 3) + (threadIdx.x >> 5);
    while(w_start < C_colIdx_size) {
        int w_C_col = C_colIdx[w_start];
        int w_C_row = C_rowIdx[w_start];
        decltype(A_colIdx) A_colIdx_segment = A_colIdx + A_rowPtr[w_C_row];
        decltype(B_rowIdx) B_rowIdx_segment = B_rowIdx + B_colPtr[w_C_col];
        int A_colIdx_segment_len = A_rowPtr[w_C_row + 1] - A_rowPtr[w_C_row];
        int B_rowIdx_segment_len = B_colPtr[w_C_col + 1] - B_colPtr[w_C_col];
        int AorB = A_colIdx_segment_len <= B_rowIdx_segment_len ? 1 : 0; // A = 1; B = 0;

        if constexpr(pass == 0) 
        {
        int curr_count = 0;
        if(AorB)
        curr_count = __find_pairs<0>(warp, nullptr, nullptr, C_targetTile, w_start, A_colIdx_segment, A_colIdx_segment_len, B_rowIdx_segment, B_rowIdx_segment_len, A_rowPtr[w_C_row], B_colPtr[w_C_col], AorB, B_tileOffsets, 0);
        else
        curr_count = __find_pairs<0>(warp, nullptr, nullptr, C_targetTile, w_start, B_rowIdx_segment, B_rowIdx_segment_len, A_colIdx_segment, A_colIdx_segment_len, B_colPtr[w_C_col], A_rowPtr[w_C_row], AorB, B_tileOffsets, 0);
        
        cg::reduce_store_async(warp, &pairs_insertion_offset[w_start], curr_count, cg::plus<int>());
        }
        else 
        {
        if(AorB)
        __find_pairs<1>(warp ,pairs_a, pairs_b, C_targetTile, w_start, A_colIdx_segment, A_colIdx_segment_len, B_rowIdx_segment, B_rowIdx_segment_len, A_rowPtr[w_C_row], B_colPtr[w_C_col], AorB, B_tileOffsets, pairs_insertion_offset[w_start]);
        else
        __find_pairs<1>(warp, pairs_a, pairs_b, C_targetTile, w_start, B_rowIdx_segment, B_rowIdx_segment_len, A_colIdx_segment, A_colIdx_segment_len, B_colPtr[w_C_col], A_rowPtr[w_C_row], AorB, B_tileOffsets, pairs_insertion_offset[w_start]);
        }
        
        w_start += (gridDim.x << 3);
    }
}

enum TAGS {
    // ONE,
    // SIXTEEN,
    NONDENSE,
    DENSE,
    TAGS_COUNT
};

template<typename ValueType, int tileSize = 16>
__global__ void __launch_bounds__(4 * 32) 
allocate_C_noshmem
(
    cr_Ptr<int> d_pairs_a,
    cr_Ptr<int> d_pairs_b,
    int d_pairs_size,

    int Ctiles_size,
    r_Ptr<int> _C_perTileNnz,
    cr_Ptr<int> C_targetTiles,

    cr_Ptr<TileCSR_rev<ValueType,tileSize>> Atiles,
    cr_Ptr<TileCSR_rev<ValueType,tileSize>> Btiles,
    cr_Ptr<uint16_t> Btiles_transposed_mask,

    cr_Ptr<int> pairs_insertion_offset,
    r_Ptr<unsigned> Ctiles_mask
)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    auto isZero = [](unsigned n) __attribute__((always_inline)) { return ((n | (~n + 1)) >> 31) & 1; };
    auto wt_u32 = []<typename T>(uint64_t global, T val) __attribute__((always_inline)) { asm volatile("st.global.L1::evict_first.u32 [%0], %1;" : : "l"(global) , "r"(val)); };

    int quarter_mgr = threadIdx.x >> 3;
    int quarter_tr = threadIdx.x % 8;
    int quarter_group_C_idx = (blockIdx.x << 4) + (threadIdx.x >> 3);
    auto quarter_group = cg::tiled_partition<8>(warp);
    while(quarter_group_C_idx < Ctiles_size)
    {
        unsigned quarter_tr_Ctiles_mask = 0;
        int quarter_local_group_pairs_count = pairs_insertion_offset[quarter_group_C_idx+1]-pairs_insertion_offset[quarter_group_C_idx];

        for(int pair = pairs_insertion_offset[quarter_group_C_idx]; pair < pairs_insertion_offset[quarter_group_C_idx] + quarter_local_group_pairs_count; ++pair)
        {
            int quarter_local_group_tile_idx_A = d_pairs_a[pair];
            int quarter_local_group_tile_idx_B = d_pairs_b[pair];

            unsigned C_mask = 0;
            #pragma unroll
            for(int n = 0; n < tileSize; ++n) C_mask |= (isZero((Atiles[quarter_local_group_tile_idx_A].mask[quarter_tr<<1] & Btiles_transposed_mask[(quarter_local_group_tile_idx_B<<4) + n])) << n);
            C_mask <<= 16;
            #pragma unroll
            for(int n = 0; n < tileSize; ++n) C_mask |= (isZero((Atiles[quarter_local_group_tile_idx_A].mask[(quarter_tr<<1)+1] & Btiles_transposed_mask[(quarter_local_group_tile_idx_B<<4) +n])) << n);

            quarter_tr_Ctiles_mask |= C_mask;
        }

        wt_u32((uint64_t)&Ctiles_mask[(quarter_group_C_idx<<3)+quarter_tr], quarter_tr_Ctiles_mask);

        int tileC_nnz = cg::reduce(quarter_group, __popc(quarter_tr_Ctiles_mask), cg::plus<int>());
        cg::invoke_one(quarter_group, wt_u32, (uint64_t)(_C_perTileNnz + quarter_group_C_idx), tileC_nnz);

        quarter_group_C_idx += (gridDim.x << 4);
    }
}

template<typename ValueType, int tileSize = 16>
__global__ void __launch_bounds__(4 * 32) 
C_setOffsets
(
    int Ctiles_size,
    r_Ptr<int> _C_perTileNnz,
    
    r_Ptr<ValueType> Ctiles_vals,
    r_Ptr<uint8_t> Ctiles_rowColIdx,
    cr_Ptr<unsigned> Ctiles_mask
)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    auto local_group = cg::tiled_partition<16>(warp);
    int lgmgr = local_group.meta_group_rank();
    int lgtr = local_group.thread_rank();

    int local_group_Ctiles_idx_start = (grid.block_rank() << 3) + (warp.meta_group_rank() << 1) + lgmgr;
    while(local_group_Ctiles_idx_start < Ctiles_size) 
    {
        unsigned my_Cmask = Ctiles_mask[(local_group_Ctiles_idx_start<<3) + (lgtr>>1)];
        my_Cmask >>= (16 * ((lgtr % 2) ^ 0x1));
        my_Cmask &= 0xFFFF;
            int Ctile_offset = _C_perTileNnz[local_group_Ctiles_idx_start];


        int my_offset = cg::exclusive_scan(local_group, __popc(my_Cmask));

        for(int n = 1; n <= __popc(my_Cmask); ++n)
        {
            unsigned c = __fns(my_Cmask, 0, n);
            unsigned my_rowColIdx = (lgtr << 4) | c;
            Ctiles_rowColIdx[Ctile_offset + (my_offset++)] = my_rowColIdx;
        }

        local_group_Ctiles_idx_start += (grid.num_blocks() << 3);
    }
}

template<typename ValueType, int tileSize = 16>
__global__ void 
__launch_bounds__(tileSize * tileSize / 2)
multiply_pairs_default
(
    cr_Ptr<int> d_pairs_a,
    cr_Ptr<int> d_pairs_b,
    
    int Ctiles_size,
    r_Ptr<ValueType> Ctiles_vals,
    r_Ptr<int> _C_perTileNnz,
    cr_Ptr<int> C_targetTiles,
    cr_Ptr<uint8_t> Ctiles_rowColIdx,

    cr_Ptr<TileCSR_rev<ValueType,tileSize>> Atiles,
    cr_Ptr<TileCSR_rev<ValueType,tileSize>> Btiles,

    cr_Ptr<uint16_t> Btiles_transposed_mask,
    cr_Ptr<int> C_targetTiles_offset
)
{
    using IdxType = uint8_t;
    __shared__ IdxType warp_tileC_rowColIdx [4][256];

    int volatile warp_tr = threadIdx.x % 32;
    int volatile warp_mgr = threadIdx.x >> 5;
    int volatile halfwarp_tr = (threadIdx.x%32)%16;
    int volatile halfwarp_mgr = (threadIdx.x%32)>>4;
    auto volatile halfwarp_tile = halfwarp_mgr == 0 ? Atiles : Btiles;    
    auto volatile halfwarp_pairs = halfwarp_mgr == 0 ? d_pairs_a : d_pairs_b;

    int warp_tileC_idx = (blockIdx.x << 2) + warp_mgr;
    while(warp_tileC_idx < Ctiles_size)
    {
        int volatile tileC_offset = _C_perTileNnz[warp_tileC_idx];
        int volatile tileC_nnz = _C_perTileNnz[warp_tileC_idx + 1] - tileC_offset;
        int volatile d_pairs_count = C_targetTiles_offset[warp_tileC_idx+1] - C_targetTiles_offset[warp_tileC_idx];

        for(int i = warp_tr; i < tileC_nnz; i += 32)
        {
            warp_tileC_rowColIdx[warp_mgr][i] = Ctiles_rowColIdx[tileC_offset + i];
        }

        for(int pair = C_targetTiles_offset[warp_tileC_idx]; pair < C_targetTiles_offset[warp_tileC_idx] + d_pairs_count; ++pair) 
        {
            int A = d_pairs_a[pair];
            int B = d_pairs_b[pair];

            // calculate C
            for(int n = warp_tr; n < tileC_nnz; n+=32)
            {
                ValueType sum = 0;
                int r = warp_tileC_rowColIdx[warp_mgr][n] >> 4;
                int c = warp_tileC_rowColIdx[warp_mgr][n] & 0xF;
                unsigned my_mask = Atiles[A].mask[r] & Btiles_transposed_mask[(B<<4)+c];
                while(my_mask)
                {
                    int ffs = __ffs(my_mask)-1;
                    int A_offset = __popc( Atiles[A].mask[r] & (0xFFFFU >> (16-ffs)) );
                    int B_offset = __popc( Btiles[B].mask[ffs] & (0xFFFFU >> (16-c)) );
                    sum += Atiles[A].vals[Atiles[A].rowPtr[r]+A_offset] * Btiles[B].vals[Btiles[B].rowPtr[ffs]+B_offset];

                    my_mask &= (~(1 << (ffs)));
                }
                Ctiles_vals[tileC_offset + n] += sum;
            }
        }
        warp_tileC_idx += (gridDim.x << 2);
    }
}

template<typename ValueType, int tileSize = 16>
__global__ void __launch_bounds__(tileSize * tileSize)
sanitize_C
(
    r_Ptr<int> rows, 
    r_Ptr<int> cols, 
    // r_Ptr<ValueType> vals,
    cr_Ptr<uint8_t> Ctiles_rowColIdx,
    cr_Ptr<ValueType> Ctiles_vals,
    int Ctiles_size,
    cr_Ptr<int> _C_tileRowIdx,
    cr_Ptr<int> _C_tileColIdx, 
    cr_Ptr<int> _C_perTile_Nnz
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
            // ValueType t_val = Ctiles_vals[warp_Ctile_offset + n];
            rows[idx] = (warp_Ctile_y<<4) + (t_rowColIdx>>4);
            cols[idx] = (warp_Ctile_x<<4) + (t_rowColIdx&0xF);
            // vals[idx] = t_val;
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

int main(int argc, char *argv[]) {
    if(argc <= 1 || argc > 4) {
        std::cout << "Provide matrix market file path for A and B (or TRANSPOSE for A * At). Exiting\n";
        exit(1);
    }

    constexpr int tileSize = 16;
    using ValueType = double;

    std::array<cudaStream_t, STREAMS_COUNT> streams;
    std::for_each(streams.begin(), streams.end(), [](cudaStream_t &s){cudaStreamCreate(&s);});

    // HOST MATRIX A ----------------------
    thrustHvecPin<int> A_I, A_J;
    thrustHvecPin<ValueType> A_val;
    int A_rows, A_cols, A_nnz;
    //-------------------------------------

    // HOST MATRIX B ----------------------
    thrustHvecPin<int> B_I, B_J;
    thrustHvecPin<ValueType> B_val;
    int B_rows, B_cols, B_nnz;
    //-------------------------------------

    std::jthread read_A(read_matrix_market<ValueType>, std::ref(argv[1]), std::ref(A_I), std::ref(A_J), std::ref(A_val), std::ref(A_rows), std::ref(A_cols), std::ref(A_nnz));
    // read_matrix_market(argv[1], A_I, A_J, A_val, A_rows, A_cols, A_nnz);
    read_matrix_market(argv[1], B_I, B_J, B_val, B_rows, B_cols, B_nnz);
    read_A.join();

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
    auto SPGEMM_STREAM_ALLOCATOR_FLOAT = [&SPGEMM_MR](cudaStream_t STREAM) {return rmm::mr::thrust_allocator<float>(STREAM, SPGEMM_MR);};
    auto SPGEMM_STREAM_ALLOCATOR_LONGLONG = [&SPGEMM_MR](cudaStream_t STREAM) {return rmm::mr::thrust_allocator<long long>(STREAM, SPGEMM_MR);};
    auto SPGEMM_STREAM_ALLOCATOR_VALUETYPE = [&SPGEMM_MR](cudaStream_t STREAM) {return rmm::mr::thrust_allocator<ValueType>(STREAM, SPGEMM_MR);};
    auto SPGEMM_STREAM_ALLOCATOR_TILECSR_REV = [&SPGEMM_MR](cudaStream_t STREAM) {return rmm::mr::thrust_allocator<TileCSR_rev<ValueType>>(STREAM, SPGEMM_MR);};
    auto SPGEMM_STREAM_ALLOCATOR_TILECSRC_REV = [&SPGEMM_MR](cudaStream_t STREAM) {return rmm::mr::thrust_allocator<TileCSR_C_rev<ValueType>>(STREAM, SPGEMM_MR);};
    auto SPGEMM_STREAM_ALLOCATOR_UINT8 = [&SPGEMM_MR](cudaStream_t STREAM) {return rmm::mr::thrust_allocator<uint8_t>(STREAM, SPGEMM_MR);};
    auto SPGEMM_STREAM_ALLOCATOR_UINT16 = [&SPGEMM_MR](cudaStream_t STREAM) {return rmm::mr::thrust_allocator<uint16_t>(STREAM, SPGEMM_MR);};
    auto SPGEMM_STREAM_ALLOCATOR_HALF = [&SPGEMM_MR](cudaStream_t STREAM) {return rmm::mr::thrust_allocator<half>(STREAM, SPGEMM_MR);};


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

    thrust::copy(ASYNC_EXEC_POLICY(STREAM_A), A_I.begin(), A_I.end(), A_d_I.begin());
    thrust::copy(ASYNC_EXEC_POLICY(STREAM_A), A_J.begin(), A_J.end(), A_d_J.begin());
    thrust::copy(ASYNC_EXEC_POLICY(STREAM_A), A_val.begin(), A_val.end(), A_d_val.begin());


    if(argc == 4 && argv[3])
    {
        std::cout << "TRANSPOSING B\n";
        thrust::copy(ASYNC_EXEC_POLICY(STREAM_B), B_I.begin(), B_I.end(), B_d_J.begin());
        thrust::copy(ASYNC_EXEC_POLICY(STREAM_B), B_J.begin(), B_J.end(), B_d_I.begin());
        thrust::copy(ASYNC_EXEC_POLICY(STREAM_B), B_val.begin(), B_val.end(), B_d_val.begin());
    }
    else
    {
        thrust::copy(ASYNC_EXEC_POLICY(STREAM_B), B_I.begin(), B_I.end(), B_d_I.begin());
        thrust::copy(ASYNC_EXEC_POLICY(STREAM_B), B_J.begin(), B_J.end(), B_d_J.begin());
        thrust::copy(ASYNC_EXEC_POLICY(STREAM_B), B_val.begin(), B_val.end(), B_d_val.begin());
    }

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

    cudaEvent_t A_tileConversion_start, A_tileConversion_end, B_tileConversion_start, B_tileConversion_end;
    cudaEvent_t high_level_multiply_start, high_level_multiply_end;
    cudaEvent_t cusparse_start, cusparse_end;
    cudaEvent_t allocate_c_start, sp0_end, sp1_start, sp1_end, aC_start, aC_end, setOffset_start, allocate_c_end;
    cudaEvent_t accumulator_end;

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
    // cudaEventCreate(&accumulator_start);
    cudaEventCreate(&accumulator_end);

    dim3 A_threads_gtc {tileSize * tileSize};
    dim3 A_blocks_gtc {cntA};

    rmm::device_vector<TileCSR_rev<ValueType>> Atiles(cntA, SPGEMM_STREAM_ALLOCATOR_TILECSR_REV(STREAM_A));
    rmm::device_vector<ValueType> Atiles_vals(A_nnz, SPGEMM_STREAM_ALLOCATOR_VALUETYPE(STREAM_A));
    rmm::device_vector<uint8_t> Atiles_rowColIdx(A_nnz, SPGEMM_STREAM_ALLOCATOR_UINT8(STREAM_A));
    rmm::device_vector<int> 
    A_d_cols(1, A_cols, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_A)), 
    A_participating_tiles_size(1, cntA, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_A));

    cudaEventRecord(A_tileConversion_start, STREAM_A);
    generate_tiles_csr<ValueType><<<A_blocks_gtc, A_threads_gtc,0, STREAM_A>>>
    (
        Atiles.data().get(), 
        Atiles_vals.data().get(),
        Atiles_rowColIdx.data().get(),
        A_participating_tiles.data().get(), 
        A_participating_tiles_size.data().get(), 
        A_perTileNnz.data().get(), 
        A_d_J.data().get(), 
        A_d_val.data().get(), 
        A_d_rowPtr.data().get(),
        A_rows
    );
    cudaEventRecord(A_tileConversion_end, STREAM_A);

    dim3 B_threads_gtc {tileSize * tileSize};
    dim3 B_blocks_gtc {cntB};

    rmm::device_vector<TileCSR_rev<ValueType>> Btiles(cntB, SPGEMM_STREAM_ALLOCATOR_TILECSR_REV(STREAM_B));
    rmm::device_vector<ValueType> Btiles_vals(B_nnz, SPGEMM_STREAM_ALLOCATOR_VALUETYPE(STREAM_B));
    rmm::device_vector<uint8_t> Btiles_rowColIdx(B_nnz, SPGEMM_STREAM_ALLOCATOR_UINT8(STREAM_B));
    rmm::device_vector<int> 
    B_d_cols(1, B_cols, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B)), 
    B_participating_tiles_size(1, cntB, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B));

    cudaEventRecord(B_tileConversion_start, STREAM_B);
    generate_tiles_csr<ValueType><<<B_blocks_gtc, B_threads_gtc,0, STREAM_B>>>
    (
        Btiles.data().get(),
        Btiles_vals.data().get(),
        Btiles_rowColIdx.data().get(),
        B_participating_tiles.data().get(), 
        B_participating_tiles_size.data().get(), 
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
    __transpose_B_mask<<<blocks_tBm, threads_tBm, 0, STREAM_B>>>(Btiles_transposed_mask.data().get(), Btiles.data().get(), cntB);

    // create High level Representation of A -> A_
    rmm::device_vector<int> _A_tileRowPtr_tmp(A_tileRows, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_D));
    rmm::device_vector<int> _A_tileRowPtr(A_tileRows + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_D));
    rmm::device_vector<int> _A_tileColIdx(cntA, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_D));
    rmm::device_vector<half> _A_tileVals(cntA, SPGEMM_STREAM_ALLOCATOR_HALF(STREAM_D));
    
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

    // create High level Representation of B -> B_
    rmm::device_vector<int> _B_tileRowPtr_tmp(B_tileRows, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    rmm::device_vector<int> _B_tileRowPtr(B_tileRows + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    rmm::device_vector<int> _B_tileColIdx(cntB, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    rmm::device_vector<half> _B_tileVals(cntB, SPGEMM_STREAM_ALLOCATOR_HALF(STREAM_E));
    
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
    
    // transpose _B
    rmm::device_vector<int> _B_tileOffsets(cntB, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    thrust::sequence(ASYNC_EXEC_POLICY(STREAM_E), _B_tileOffsets.begin(), _B_tileOffsets.end());
    thrust::transform(ASYNC_EXEC_POLICY(STREAM_E), B_participating_tiles.begin(), B_participating_tiles.end(), B_participating_tiles.begin(), swap32());
    {
    auto zit = thrust::make_zip_iterator(thrust::make_tuple(B_participating_tiles.begin(), _B_tileOffsets.begin()));
    thrust::sort(ASYNC_EXEC_POLICY(STREAM_E), zit, zit+B_participating_tiles.size());
    }

    rmm::device_vector<int> _B_tileColPtr_tmp(B_tileCols, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    rmm::device_vector<int> _B_tileColPtr(B_tileCols + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    rmm::device_vector<int> _B_tileRowIdx(cntB, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
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

    cudaStreamSynchronize(STREAM_D);
    rmm::device_vector<long long>().swap(A_participating_tiles);
    rmm::device_vector<int>().swap(_A_tileRowPtr_tmp);
    cudaStreamSynchronize(STREAM_E);
    rmm::device_vector<long long>().swap(B_participating_tiles);
    rmm::device_vector<int>().swap(_B_tileRowPtr_tmp);

    cudaStreamWaitEvent(STREAM_C, A_tileConversion_end);
    cudaStreamWaitEvent(STREAM_C, B_tileConversion_end);

    int *_C_tileColIdx, *_C_tileRowIdx;

    std::chrono::high_resolution_clock::time_point pem_spgemm_start;
    // cudaEventRecord(high_level_multiply_start, STREAM_C);
    int _C_nnz = cusparse_highLevelMultiply
    (
        _A_tileRowPtr.data().get(),
        _A_tileColIdx.data().get(),
        _A_tileVals.data().get(),
        A_tileRows,
        A_tileCols,
        _A_tileVals.size(),
        _B_tileRowPtr.data().get(),
        _B_tileColIdx.data().get(),
        _B_tileVals.data().get(),
        B_tileRows,
        B_tileCols,
        _B_tileVals.size(),
        &_C_tileColIdx, &_C_tileRowIdx,
        STREAM_C,
        ASYNC_EXEC_POLICY(STREAM_C),
        cusparse_start,
        cusparse_end,
        high_level_multiply_start,
        high_level_multiply_end,
        pem_spgemm_start
    );
    // cudaEventRecord(high_level_multiply_end, STREAM_C);
    
    std::cout << "\nCOUNTING PAIRS\n";
    dim3 threads_sp{256};
    dim3 blocks_sp{(_C_nnz-1+threads_sp.x)/threads_sp.x};

    rmm::device_vector<int> pairs_insertion_offset(_C_nnz+1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_C));

    cudaEventRecord(allocate_c_start, STREAM_C);
    search_pairs<0><<<blocks_sp, threads_sp, 0, STREAM_C>>>
    (
        nullptr,
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
        pairs_insertion_offset.data().get()
    );
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_C), pairs_insertion_offset.begin(), pairs_insertion_offset.end(), pairs_insertion_offset.begin());
    cudaEventRecord(sp0_end, STREAM_C);

    int *d_pairs_a, *d_pairs_b, *C_targetTile;
    cudaMallocAsync(&d_pairs_a, sizeof(int) * pairs_insertion_offset.back(), STREAM_C); 
    cudaMallocAsync(&d_pairs_b, sizeof(int) * pairs_insertion_offset.back(), STREAM_C); 
    cudaMallocAsync(&C_targetTile, sizeof(int) * pairs_insertion_offset.back(), STREAM_C); 


    cudaEventRecord(sp1_start, STREAM_C);
    search_pairs<1><<<blocks_sp, threads_sp, 0, STREAM_C>>>
    (
        d_pairs_a,
        d_pairs_b,
        C_targetTile,
        _C_tileColIdx,
        _A_tileRowPtr.data().get(),
        _A_tileColIdx.data().get(),
        _B_tileColPtr.data().get(),
        _B_tileRowIdx.data().get(),
        A_tileRows,
        _C_nnz,
        _B_tileOffsets.data().get(),
        _C_tileRowIdx,
        pairs_insertion_offset.data().get()
    );
    cudaEventRecord(sp1_end, STREAM_C);

    unsigned *Ctiles_mask;
    cudaMallocAsync(&Ctiles_mask, sizeof(unsigned) * 8 * _C_nnz, STREAM_C);
    rmm::device_vector<int> _C_perTileNnz(_C_nnz + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_C));

    dim3 threads_aC {128};
    dim3 blocks_aC {(_C_nnz+15)/16};

    cudaEventRecord(aC_start, STREAM_C);
    allocate_C_noshmem<<<blocks_aC, threads_aC, 0, STREAM_C>>>
    (
        d_pairs_a,
        d_pairs_b,
        pairs_insertion_offset.back(),
        _C_nnz,
        _C_perTileNnz.data().get(),
        C_targetTile,
        Atiles.data().get(),
        Btiles.data().get(),
        Btiles_transposed_mask.data().get(),
        pairs_insertion_offset.data().get(),
        Ctiles_mask
    );

    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_C), _C_perTileNnz.begin(), _C_perTileNnz.end(), _C_perTileNnz.begin());
    cudaEventRecord(aC_end, STREAM_C);

    ValueType *Ctiles_vals;
    uint8_t *Ctiles_rowColIdx;
    cudaMallocAsync(&Ctiles_vals, sizeof(ValueType) * _C_perTileNnz.back(), STREAM_C);
    cudaMallocAsync(&Ctiles_rowColIdx, sizeof(uint8_t) * _C_perTileNnz.back(), STREAM_C);

    dim3 threads_Cs{128};
    dim3 blocks_Cs{(_C_nnz+7)/8};

    cudaEventRecord(setOffset_start, STREAM_C);
    C_setOffsets<<<blocks_Cs, threads_Cs, 0, STREAM_C>>>
    (
        _C_nnz,
        _C_perTileNnz.data().get(),
        Ctiles_vals,
        Ctiles_rowColIdx,
        Ctiles_mask
    );
    cudaEventRecord(allocate_c_end, STREAM_C);

    std::cout << "\nACCUMULATOR PHASE\n\n\n";

    dim3 threads_mp {128};
    dim3 blocks_mp {(_C_nnz+3)/4};

    // cudaEventRecord(accumulator_start, STREAM_C);
    multiply_pairs_default<ValueType><<<blocks_mp, threads_mp, 0, STREAM_C>>>
    (
        d_pairs_a,
        d_pairs_b,
        _C_nnz,
        Ctiles_vals,
        _C_perTileNnz.data().get(),
        C_targetTile,
        Ctiles_rowColIdx,
        Atiles.data().get(),
        Btiles.data().get(),
        Btiles_transposed_mask.data().get(),
        pairs_insertion_offset.data().get()
    );
    cudaEventRecord(accumulator_end, STREAM_C);

    cudaDeviceSynchronize();
    auto pem_spgemm_end = std::chrono::high_resolution_clock::now();
    auto pem_spgemm_duration = std::chrono::duration<double, std::milli>(pem_spgemm_end-pem_spgemm_start);
    
    float Aconversion, Bconversion;
    cudaEventElapsedTime(&Aconversion, A_tileConversion_start, A_tileConversion_end);
    cudaEventElapsedTime(&Bconversion, B_tileConversion_start, B_tileConversion_end);

    float hlm_time, cusparse_time, sp0_time, sp1_time, aC_time, sO_time, accumulator_time;
    cudaEventElapsedTime(&hlm_time, high_level_multiply_start, high_level_multiply_end);
    cudaEventElapsedTime(&cusparse_time, cusparse_start, cusparse_end);
    // cudaEventElapsedTime(&allocate_c_time, allocate_c_start, allocate_c_end);
    cudaEventElapsedTime(&sp0_time, allocate_c_start, sp0_end);
    cudaEventElapsedTime(&sp1_time, sp1_start, sp1_end);
    cudaEventElapsedTime(&aC_time, aC_start, aC_end);
    cudaEventElapsedTime(&sO_time, setOffset_start, allocate_c_end);
    cudaEventElapsedTime(&accumulator_time, allocate_c_end, accumulator_end);

    float allocate_c_time = sp0_time + sp1_time + aC_time + sO_time;

    float pem_spgemm_time = pem_spgemm_duration.count();
    float kernel_time = hlm_time + allocate_c_time + accumulator_time;
    float malloc_time = pem_spgemm_time - kernel_time;
    
    std::cout << "<---Program done--->\n";
    std::cout << "Matrix A CSR to tile conversion took " << Aconversion << "ms\n";
    std::cout << "Matrix B CSR to tile conversion took " << Bconversion << "ms\n";
    std::cout << "total--------------------------------" << Aconversion+Bconversion << "ms\n\n";

    std::cout << "High-Level tile multiplication took--" << hlm_time << "ms\n";
    std::cout << "*cusparse_hlm took------" << cusparse_time << "ms\n";
    std::cout << "Allocating C took--------------------" << allocate_c_time << "ms\n";
    std::cout << "Accumulator took---------------------" << accumulator_time << "ms\n\n";
    
    std::cout << "PEM-SPGEMM took " << std::fixed << std::setprecision(2) 
    << pem_spgemm_time << "ms\nKernel time " << kernel_time << "ms\nmalloc time " << malloc_time << "ms\n";
    std::cout << "C tiles: " << _C_nnz << "\n";
    std::cout << "C nnz: " << _C_perTileNnz.back() << "\n";

    if(!atoi(argv[2]))
    {
        std::cout << "Not saving results. Exiting.\n";
        std::atexit([]{cudaDeviceReset();});
        return 0;
    }

    rmm::device_vector<int>(SPGEMM_STREAM_ALLOCATOR_INT(STREAM_C)).swap(pairs_insertion_offset);
    cudaFreeAsync(d_pairs_a, STREAM_C);
    cudaFreeAsync(d_pairs_b, STREAM_C);
    cudaFreeAsync(C_targetTile, STREAM_C);

    rmm::device_vector<int> Crows(_C_perTileNnz.back(), SPGEMM_STREAM_ALLOCATOR_INT(STREAM_C));
    rmm::device_vector<int> Ccols(_C_perTileNnz.back(), SPGEMM_STREAM_ALLOCATOR_INT(STREAM_C));
    // rmm::device_vector<ValueType> Cvals(_C_perTileNnz.back(), SPGEMM_STREAM_ALLOCATOR_VALUETYPE(STREAM_C));

    dim3 threads_sC {tileSize * tileSize};
    dim3 blocks_sC {(_C_nnz+7)/8};

    cudaEvent_t sanitize_C_start, sanitize_C_end;
    cudaEventCreate(&sanitize_C_start);
    cudaEventCreate(&sanitize_C_end);

    cudaEventRecord(sanitize_C_start, STREAM_C);
    sanitize_C<<<blocks_sC, threads_sC, 0, STREAM_C>>>
    (
        Crows.data().get(), 
        Ccols.data().get(), 
        // Cvals.data().get(), 
        Ctiles_rowColIdx,
        Ctiles_vals,
        _C_nnz,
        _C_tileRowIdx,
        _C_tileColIdx, 
        _C_perTileNnz.data().get()
    );
    cudaEventRecord(sanitize_C_end, STREAM_C);

    {
        auto zit = thrust::make_zip_iterator(Crows.begin(), Ccols.begin(), Ctiles_vals);
        thrust::stable_sort(ASYNC_EXEC_POLICY(STREAM_C), zit, zit + _C_perTileNnz.back());
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
    auto print2 = [&outfile] (auto v) { outfile << v << " "; };
    
    thrustHvec<int> hCrows(Crows.size());
    thrustHvec<int> hCcols(Ccols.size());
    std::vector<ValueType> hCvals(_C_perTileNnz.back());

    thrust::copy(Crows.begin(), Crows.end(), hCrows.begin());
    thrust::copy(Ccols.begin(), Ccols.end(), hCcols.begin());
    cudaMemcpy(hCvals.data(), Ctiles_vals, sizeof(ValueType) * _C_perTileNnz.back(), cudaMemcpyDeviceToHost);


    outfile.open(filename0, std::ios::out);
    outfile << _C_perTileNnz.back();
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

    std::atexit([]{cudaDeviceReset();});
    return 0; // <--------------------------------------------------------------------------------------------------------------------------------

    // streams are destroyed by rmm
}
