/*
Author: Petrus E. Manurung
*/

#include <cstdlib>
#include <cstdio>

#include <fstream>
#include <thread>
#include <algorithm>

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/unique.h>
#include <thrust/partition.h>
#include <thrust/iterator/transform_iterator.h>

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
    int tileRows,
    int nnz
)
{
    auto grid = cg::this_grid();
    auto my_tid = grid.thread_rank();
    // out of range check
    if(grid.thread_rank() >= nnz) return;
    int my_y = d_I[grid.thread_rank()];
    int my_x = d_J[grid.thread_rank()];
    int this_tile_y = my_y / tileSize;
    int this_tile_x = my_x / tileSize;

    long long my_tile = (static_cast<long long>(this_tile_y) << 32) | this_tile_x;
    // long long *my_participation = &participating_tiles[this_tile_y * tileRows + this_tile_x];
    long long *my_participation = &participating_tiles[grid.thread_rank()];

    *my_participation = my_tile;
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
    // r_Ptr<int> tilePtr,
    // r_Ptr<int> tileColIdx,
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
    auto tid = block.thread_rank();

    unsigned block_id = grid.block_rank();
    while(block_id < *participating_tiles_size) {
        long long block_tile = participating_tiles[block_id];
        int block_tile_x = *(reinterpret_cast<int*>(&block_tile));
        int block_tile_y = *(reinterpret_cast<int*>(&block_tile)+1);

        int block_tile_offset_x = block_tile_x * tileSize;
        int block_tile_offset_y = block_tile_y * tileSize;
        int block_d_tiles_offset = block_id;

        ValueType thread_val {};
        int thread_J = 0;

        __shared__ int temp_buffer[16];

        auto my_row_group = cg::tiled_partition<16>(block); // swap 16 to tileSize
        
        int my_row_group_rowPtr_offset = block_tile_offset_y + my_row_group.meta_group_rank();
        if(my_row_group_rowPtr_offset >= d_rowPtr_size) return;

        int my_row_group_rowPtr = d_rowPtr[my_row_group_rowPtr_offset];
        int my_row_group_rowSize = d_rowPtr[my_row_group_rowPtr_offset + 1] - d_rowPtr[my_row_group_rowPtr_offset];

        int thread_offset = binarySearch(&d_J[my_row_group_rowPtr], block_tile_offset_x + (int)my_row_group.thread_rank(), my_row_group_rowSize);
        if(thread_offset != -1){
            thread_offset += my_row_group_rowPtr;
            thread_J = d_J[thread_offset];
            thread_val = d_vals[thread_offset];
        }
        my_row_group.sync();

        IdxType my_RowColIdx = (static_cast<IdxType>(my_row_group.meta_group_rank()) << 4) | (thread_J%16);
        unsigned my_row_nnz = __popc(my_row_group.ballot(thread_offset != -1));
        MaskType my_row_mask = thread_offset != -1 ? 1 : 0;

        my_row_mask <<= (thread_J%tileSize); // maybe optimize later with cg::reduce?
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
            temp_buffer[0] = 0; // reuse temp_buffer[0] as a counter
            d_tiles[block_d_tiles_offset].vals = d_tiles_vals + tile_offset;
            d_tiles[block_d_tiles_offset].rowColIdx = d_tiles_rowColIdx + tile_offset;
        }
        block.sync();

        for(int t = 0; t < block.size(); ++t) {
            if(tid == t && thread_val != 0) {
                d_tiles[block_d_tiles_offset].vals[temp_buffer[0]] = thread_val;
                d_tiles[block_d_tiles_offset].rowColIdx[temp_buffer[0]] = my_RowColIdx;
                // d_tiles_vals[tile_offset + temp_buffer[0]] = thread_val;
                // d_tiles_rowColIdx[tile_offset + temp_buffer[0]] = my_RowColIdx;
                ++temp_buffer[0];
            }
            block.sync();
        }
        

        block_id += grid.num_blocks();
    }
}

__attribute__((optimize("O3")))
int cusparse_highLevelMultiply
(
    cr_Ptr<int> dA_csrOffsets,
    cr_Ptr<int> dA_columns,
    cr_Ptr<float> dA_values,
    unsigned A_num_rows,
    unsigned A_num_cols,
    unsigned A_nnz,
    cr_Ptr<int> dB_csrOffsets,
    cr_Ptr<int> dB_columns,
    cr_Ptr<float> dB_values,
    unsigned B_num_rows,
    unsigned B_num_cols,
    unsigned B_nnz,
    rmm::device_vector<int> *d_CtilePtr,
    rmm::device_vector<int> *d_CtileColIdx,
    cudaStream_t stream,
    rmm::exec_policy_nosync ASYNC_EXEC_POLICY
)
{
    int *dC_csrOffsets;
    int *dC_columns;
    float *dC_values;

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;

    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;
    float               alpha       = 1.0f;
    float               beta        = 0.0f;

    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;

    CHECK_CUSPARSE( cusparseCreate(&handle) )
    CHECK_CUSPARSE( cusparseSetStream(handle, stream) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      const_cast<int*>(dA_csrOffsets), const_cast<int*>(dA_columns), const_cast<float*>(dA_values),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                      const_cast<int*>(dB_csrOffsets), const_cast<int*>(dB_columns), const_cast<float*>(dB_values),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                      dC_csrOffsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // allocate C offsets
    CHECK_CUDA( cudaMallocAsync((void**) &dC_csrOffsets,
                           (A_num_rows + 1) * sizeof(int), stream) )
    // CHECK_CUDA( cudaMalloc((void**) &dC_csrOffsets,
    //                        (A_num_rows + 1) * sizeof(int)) )

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
    // CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
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
    // CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

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
    CHECK_CUDA( cudaMallocAsync((void**) &dC_values,  C_nnz1 * sizeof(float), stream) )
    // CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int))   )
    // CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float)) )

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

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    auto _dCtile = thrust::device_pointer_cast(dC_csrOffsets);
    auto _dCcols = thrust::device_pointer_cast(dC_columns);

    d_CtilePtr->resize(A_num_rows + 1);
    d_CtileColIdx->resize(C_nnz1);
    thrust::copy(ASYNC_EXEC_POLICY, _dCtile, _dCtile + A_num_rows + 1, d_CtilePtr->begin());
    thrust::copy(ASYNC_EXEC_POLICY, _dCcols, _dCcols + C_nnz1, d_CtileColIdx->begin());
    // thrust::copy(_dCtile, _dCtile + A_num_rows + 1, d_CtilePtr->begin());
    // thrust::copy(_dCcols, _dCcols + C_nnz1, d_CtileColIdx->begin());

    cudaFreeAsync(dC_csrOffsets, stream);
    cudaFreeAsync(dC_columns, stream);
    cudaFreeAsync(dC_values, stream);
    // cudaFree(dC_csrOffsets);
    // cudaFree(dC_columns);
    // cudaFree(dC_values);
    return 0;
}

template<int tileSize = 16>
__global__ void __launch_bounds__(tileSize * tileSize)
_tag_rows
(
    r_Ptr<int> rowIdx,
    cr_Ptr<int> rowPtr,
    int rowPtr_size
)
{
    int global_tid = cg::this_grid().thread_rank();
    if(global_tid >= rowPtr_size - 1 || global_tid == 0) return;

    if(rowPtr[global_tid + 1] - rowPtr[global_tid] == 0) return;

    int my_offset = rowPtr[global_tid];
    rowIdx[my_offset] = 1;
}

__device__ cuda::atomic<int, cuda::thread_scope_device> pairs_idx(0);
template<int pass>
__device__ __forceinline__
// we iterate on lane_iter, search on lane_targ
int __find_pairs
(
    r_Ptr<long long> pairs,
    r_Ptr<int> C_targetTile,
    int t_start,
    cr_Ptr<int> lane_iter,
    int lane_iter_len,
    cr_Ptr<int> lane_targ,
    int lane_targ_len,
    int iter_offset,
    int targ_offset,
    int AorB,
    cr_Ptr<int> B_tileOffsets
)
{
    int __local_count = 0; // for use when pass = 0 only
    for(int i = 0; i < lane_iter_len; ++i) {
        int found = binarySearch(lane_targ, lane_iter[i], lane_targ_len);
        if(found != -1) {
            if constexpr(pass == 1) // 1 = second pass
            {   
                int first = iter_offset + i;
                int second = targ_offset + found;
                if(!AorB) // meaning its B, need to swap-- order is AB
                std::swap(first, second);
                second = B_tileOffsets[second];
                int targ_idx = pairs_idx.fetch_add(1, cuda::memory_order_relaxed);
                pairs[targ_idx] = ((static_cast<long long>(first) << 32) | second);
                C_targetTile[targ_idx] = t_start;
            }
            else // 0 = first pass, only count how many are there
            ++__local_count;
        }
    }
    return __local_count;
}

template<int pass>
__global__ void __launch_bounds__(256) 
search_pairs
(
    r_Ptr<long long> pairs,
    r_Ptr<int> C_targetTile,
    cr_Ptr<int> C_rowPtr,
    cr_Ptr<int> C_colIdx,
    cr_Ptr<int> A_rowPtr,
    cr_Ptr<int> A_colIdx,
    cr_Ptr<int> B_colPtr,
    cr_Ptr<int> B_rowIdx,
    cr_Ptr<int> C_rowPtr_size,
    cr_Ptr<int> C_colIdx_size,
    cr_Ptr<int> B_tileOffsets,
    int *__restrict__ pairs_counter
)
{
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto thread = cg::this_thread();

    int __first_pass_thread_counter = 0; // for pass = 0 only

    int t_start = grid.thread_rank();
    int grid_num_threads = grid.num_threads();
    while(t_start < *C_colIdx_size) {
        int b_C_cols = C_colIdx[t_start];

        int t_C_row = lowerBound(C_rowPtr, t_start, *C_rowPtr_size);
        int t_C_col = b_C_cols;
        decltype(A_colIdx) A_colIdx_segment = A_colIdx + A_rowPtr[t_C_row];
        decltype(B_rowIdx) B_rowIdx_segment = B_rowIdx + B_colPtr[t_C_col];
        int A_colIdx_segment_len = A_rowPtr[t_C_row + 1] - A_rowPtr[t_C_row];
        int B_rowIdx_segment_len = B_colPtr[t_C_col + 1] - B_colPtr[t_C_col];
        int AorB = A_colIdx_segment_len <= B_rowIdx_segment_len ? 1 : 0; // A = 1; B = 0;

        if constexpr(pass == 0) {
        if(AorB)
        __first_pass_thread_counter += __find_pairs<0>(pairs, C_targetTile, t_start, A_colIdx_segment, A_colIdx_segment_len, B_rowIdx_segment, B_rowIdx_segment_len, A_rowPtr[t_C_row], B_colPtr[t_C_col], AorB, B_tileOffsets);
        else
        __first_pass_thread_counter += __find_pairs<0>(pairs, C_targetTile, t_start, B_rowIdx_segment, B_rowIdx_segment_len, A_colIdx_segment, A_colIdx_segment_len, B_colPtr[t_C_col], A_rowPtr[t_C_row], AorB, B_tileOffsets);
        }
        else {
        if(AorB)
        __find_pairs<1>(pairs, C_targetTile, t_start, A_colIdx_segment, A_colIdx_segment_len, B_rowIdx_segment, B_rowIdx_segment_len, A_rowPtr[t_C_row], B_colPtr[t_C_col], AorB, B_tileOffsets);
        else
        __find_pairs<1>(pairs, C_targetTile, t_start, B_rowIdx_segment, B_rowIdx_segment_len, A_colIdx_segment, A_colIdx_segment_len, B_colPtr[t_C_col], A_rowPtr[t_C_row], AorB, B_tileOffsets);
        }

        t_start += grid.num_threads();
    }

    if constexpr(pass == 0) {
        block.sync();
        __shared__ int total;
        auto warp = cg::tiled_partition<32>(block);
        if(block.thread_rank() == 0) total = 0;
        int warp_total = cg::reduce(warp, __first_pass_thread_counter, cg::plus<int>());
        if(warp.thread_rank() == 0) atomicAdd_block(&total, warp_total);
        block.sync();
        if(block.thread_rank() == 0) atomicAdd(pairs_counter, total);

        return;
    }
    grid.sync();

    if constexpr(pass == 1)
    if(grid.thread_rank() == 0) {
        // *pairs_counter = pairs_idx.load(cuda::memory_order_relaxed);
        *pairs_counter = pairs_idx.exchange(*pairs_counter, cuda::memory_order_relaxed);
    }
}

template<typename ValueType, int tileSize = 16>
__global__ void __launch_bounds__(4 * 32) 
allocate_C
(
    cr_Ptr<long long> d_pairs,
    int d_pairs_size,

    r_Ptr<TileCSR_C_rev<ValueType, tileSize>> Ctiles,
    int Ctiles_size,
    r_Ptr<int> _C_perTileNnz,
    cr_Ptr<int> C_targetTiles,

    cr_Ptr<TileCSR_rev<ValueType,tileSize>> Atiles,
    cr_Ptr<TileCSR_rev<ValueType,tileSize>> Btiles
)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    __shared__  uint16_t tiles_mask[4 * 2 * 16];
    
    auto local_group = cg::tiled_partition<16>(warp); // meta group rank 0 -> row, 1 -> col, 0 loads from A, 1 from B
    int lgmgr = local_group.meta_group_rank();
    int lgtr = local_group.thread_rank();
    auto isZero = [](unsigned n) __attribute__((always_inline)) { return ((n | (~n + 1)) >> 31) & 1; };

    int warp_pairs_idx_start = grid.block_rank() * 4 + warp.meta_group_rank();
    int warp_tiles_mask_offset = warp.meta_group_rank() * 2 * 16;
    while(warp_pairs_idx_start < d_pairs_size) 
    {
        int warp_tile_idx = (d_pairs[warp_pairs_idx_start] >> (32 * (lgmgr^0x1))) & 0xFFFFFFFF;

        if(lgtr == 0) {
            // optimization idea: async copy
            auto tile = lgmgr ? Btiles : Atiles;
            *(reinterpret_cast<ulonglong4*>(&tiles_mask[warp_tiles_mask_offset + 16 * lgmgr])) 
            = 
            *(reinterpret_cast<ulonglong4 const*>(&tile[warp_tile_idx].mask[0]));
        }
        warp.sync();

        unsigned thread_mask = tiles_mask[warp_tiles_mask_offset + lgmgr * 16 + lgtr];

        // calculate mask from A -> B
        unsigned C_mask = 0;
        #pragma unroll
        for(int c = 0; c < tileSize; ++c) {
            unsigned col_mask = (((thread_mask >> c) & 1U) << lgtr);
            col_mask = cg::reduce(local_group, col_mask, cg::bit_or<unsigned>());
            col_mask = warp.shfl_down(col_mask, 16);

            C_mask |= ( isZero(thread_mask & col_mask) << c); // C_mask valid only on first halfwarp
        }

        if(lgmgr == 0) {
            int warp_tileC_idx = C_targetTiles[warp_pairs_idx_start];
            TileCSR_C_rev<ValueType> *tileC = Ctiles + warp_tileC_idx;

            if(lgtr % 2 == 0) C_mask <<= 16;
            C_mask |= local_group.shfl_xor(C_mask, 0x1U);
            if(lgtr % 2 == 0)
            {
                auto coalesced = cg::coalesced_threads();
                atomicOr(&tileC->mask[coalesced.thread_rank()], C_mask);
            }
        }

        warp_pairs_idx_start += grid.num_blocks() * block.num_threads() / warp.size();
    }

    grid.sync();

    int local_group_Ctiles_idx_start = grid.block_rank() * 4 * 2 + warp.meta_group_rank() * 2 + lgmgr;
    while(local_group_Ctiles_idx_start < Ctiles_size) 
    {
        unsigned my_Cmask = Ctiles[local_group_Ctiles_idx_start].mask[lgtr / 2];
        my_Cmask >>= (16 * ((lgtr % 2) ^ 0x1));
        my_Cmask &= 0xFFFF;
        int nnz = __popc(my_Cmask);
        int row_nnz = cg::exclusive_scan(local_group, nnz);
        Ctiles[local_group_Ctiles_idx_start].rowPtr[lgtr] = row_nnz;
        if(lgtr == 15) _C_perTileNnz[local_group_Ctiles_idx_start] = nnz + row_nnz;
        local_group.sync();

        local_group_Ctiles_idx_start += grid.num_blocks() * block.num_threads() / local_group.size();
    }
}

template<typename ValueType, int tileSize = 16>
__global__ void __launch_bounds__(4 * 32) 
C_setOffsets
(
    r_Ptr<TileCSR_C_rev<ValueType, tileSize>> Ctiles,
    int Ctiles_size,
    r_Ptr<int> _C_perTileNnz,
    
    r_Ptr<ValueType> Ctiles_vals,
    r_Ptr<uint8_t> Ctiles_rowColIdx
)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    auto local_group = cg::tiled_partition<16>(warp); // meta group rank 0 -> row, 1 -> col, 0 loads from A, 1 from B
    int lgmgr = local_group.meta_group_rank();
    int lgtr = local_group.thread_rank();

    int local_group_Ctiles_idx_start = grid.block_rank() * 4 * 2 + warp.meta_group_rank() * 2 + lgmgr;
    while(local_group_Ctiles_idx_start < Ctiles_size) 
    {
        unsigned my_Cmask = Ctiles[local_group_Ctiles_idx_start].mask[lgtr / 2];
        my_Cmask >>= (16 * ((lgtr % 2) ^ 0x1));
        my_Cmask &= 0xFFFF;
        cg::invoke_one(local_group, [&]{
            int Ctile_offset = _C_perTileNnz[local_group_Ctiles_idx_start];
            Ctiles[local_group_Ctiles_idx_start].rowColIdx = Ctiles_rowColIdx + Ctile_offset;
            Ctiles[local_group_Ctiles_idx_start].vals = Ctiles_vals + Ctile_offset;
        });
        local_group.sync();

        int my_offset = cg::exclusive_scan(local_group, __popc(my_Cmask));

        if(__popc(my_Cmask) > 0) 
        {
            for(int n = 1; n <= __popc(my_Cmask); ++n)
            {
                unsigned c = __fns(my_Cmask, 0, n);
                unsigned my_rowColIdx = (lgtr << 4) | c;
                Ctiles[local_group_Ctiles_idx_start].rowColIdx[my_offset++] = my_rowColIdx;
            }
        }

        local_group_Ctiles_idx_start += grid.num_blocks() * block.num_threads() / local_group.size();
    }
}

template<typename ValueType, int tileSize = 16>
__device__ 
__forceinline__
void __multiply_default_sync(
    cg::thread_block_tile<32, cg::thread_block> &warp,
    long long *__restrict__ myWarp_buffer,
    TileCSR_C_rev<ValueType> *__restrict__ tileC,
    int tileC_nnz,
    TileCSR_rev<ValueType> *__restrict__ halfwarp_tile,
    int halfwarp_tile_nnz
) 
{
    auto local_group = cg::tiled_partition<16>(warp); // meta group rank 0 -> row, 1 -> col, 0 loads from A, 1 from B
    
    int lgmgr = local_group.meta_group_rank();
    int lgtr = local_group.thread_rank();

    int both_lte_16 = warp.shfl_xor(halfwarp_tile_nnz, 0xFFFFU) <= 16 && halfwarp_tile_nnz <= 16;
    if(both_lte_16) [[likely]] // register caching technique
    {   
        ValueType my_val {};
        // uint8_t my_rowColIdx {};
        uint8_t my_a{};
        uint8_t my_b{};
        if(lgtr < halfwarp_tile_nnz) {
            my_val = halfwarp_tile->vals[lgtr];
            // my_rowColIdx = halfwarp_tile->rowColIdx[lgtr];
            my_a = ((halfwarp_tile->rowColIdx[lgtr] >> (4 * (lgmgr ^ 0x1))) & 0xFU);
            my_b = ((halfwarp_tile->rowColIdx[lgtr] >> (4 * (lgmgr))) & 0xFU);
        }
        warp.sync();

        int C_idx = 0;
        while(C_idx < tileC_nnz)
        {
            myWarp_buffer[warp.thread_rank()] = 0;
            uint8_t C_rowColIdx = tileC->rowColIdx[C_idx];

            if( my_val != 0 && my_a == ((C_rowColIdx >> (4 * (lgmgr ^ 0x1))) & 0xFU) )
            {
                auto coalesced = cg::coalesced_threads();
                auto fma_buf = reinterpret_cast<ValueType*>(myWarp_buffer);

                unsigned match_mask = ((1 << my_b) << (16 * lgmgr));
                match_mask = cg::reduce(coalesced, match_mask, cg::bit_or<unsigned>());
                match_mask = (match_mask >> 16) & (match_mask & 0xFFFF);
                int n = 0;
                if((1 << my_b) & match_mask)
                {
                    auto matched = cg::coalesced_threads();
                    n = matched.num_threads()/2;
                    fma_buf[matched.thread_rank()] = my_val;
                }
                coalesced.sync();
                if(coalesced.thread_rank() == 0)
                {
                    ValueType sum {};
                    for(int i = 0; i < n; ++i) sum += fma_buf[i] * fma_buf[n+i];
                    atomicAdd(&tileC->vals[C_idx], sum);
                }
            }
            warp.sync();
            ++C_idx;
        }
    }
    else [[unlikely]]
    {
        unsigned C_mask = ((tileC->mask[lgtr/2]) >> (16 * ((lgtr % 2) ^ 0x1))) & 0xFFFF;
        int C_curr_counter = 0;
        #pragma unroll
        for(int r = 0; r < tileSize; ++r) {
            unsigned current_Cmask = warp.shfl(C_mask, r);
            if(__popc(current_Cmask) == 0) continue;

            // ValueType my_sum {};
            ValueType my_elem {};
            
            if(lgmgr == 0) {
                myWarp_buffer[lgtr] = -1;
                int Arow_len = (r == tileSize - 1 ? halfwarp_tile_nnz : halfwarp_tile->rowPtr[r + 1]) - halfwarp_tile->rowPtr[r];

                if(lgtr < Arow_len) {
                    int local_group_tileA_offset = halfwarp_tile->rowPtr[r];
                    my_elem = halfwarp_tile->vals[local_group_tileA_offset + lgtr];
                    myWarp_buffer[halfwarp_tile->rowColIdx[local_group_tileA_offset + lgtr] & 0x0F] = lgtr;
                }
                local_group.sync();

                int lowest_thread_with_0 = __ffs(local_group.ballot(my_elem == 0)) - 1;
                if(myWarp_buffer[lgtr] == -1) myWarp_buffer[lgtr] = lowest_thread_with_0;
                my_elem = local_group.shfl(my_elem, myWarp_buffer[lgtr]);
            }
            warp.sync();

            for(int n = 1; n <= __popc(current_Cmask); ++n)
            {
                int c = __fns(current_Cmask, 0, n);
                if(lgmgr == 1) {
                    my_elem = 0; // reset
                    int thread_Brow_offset = halfwarp_tile->rowPtr[lgtr];
                    int thread_Brow_len = (lgtr == 15 ? halfwarp_tile_nnz : halfwarp_tile->rowPtr[lgtr + 1]) - halfwarp_tile->rowPtr[lgtr];

                    if(thread_Brow_len != 0) {
                        int found = binarySearch(halfwarp_tile->rowColIdx + thread_Brow_offset, (uint8_t)((lgtr << 4) | c), thread_Brow_len);
                        if(found != -1) my_elem = halfwarp_tile->vals[thread_Brow_offset + found];
                    }
                }

                if constexpr(std::is_floating_point_v<ValueType> && sizeof(ValueType) == 8)
                {
                ValueType row_sum = warp.shfl_down(my_elem, 16);
                row_sum *= my_elem;
                row_sum += local_group.shfl_down(row_sum, 1);            
                row_sum += local_group.shfl_down(row_sum, 2);      
                row_sum += local_group.shfl_down(row_sum, 4);      
                row_sum += local_group.shfl_down(row_sum, 8);      
                row_sum = warp.shfl(row_sum, 0);    
                if(lgmgr == 0 && lgtr == c) atomicAdd(&tileC->vals[C_curr_counter+(n-1)], row_sum);
                } 
                else {
                auto fma_buf = reinterpret_cast<ValueType*>(myWarp_buffer);
                *(fma_buf + warp.thread_rank()) = my_elem;
                if(lgmgr == 0 && lgtr == c) {
                    ValueType my_sum {};
                    #pragma unroll
                    for(int i = 0; i < 16; ++i) 
                    my_sum += fma_buf[i] * fma_buf[i + 16];
                    atomicAdd(&tileC->vals[C_curr_counter+(n-1)], my_sum);
                }
                }
            }
            C_curr_counter += __popc(current_Cmask);
            warp.sync();

            // store
            // if(lgmgr == 0 && my_sum != 0) atomicAdd(&tileC->vals[r * tileSize + lgtr], my_sum);
            // warp.sync();
        }
    }
}


template<typename ValueType, int tileSize = 16>
__global__ void 
__launch_bounds__(tileSize * tileSize / 2)
multiply_pairs
(
    cr_Ptr<long long> d_pairs,
    int d_pairs_size,
    
    r_Ptr<TileCSR_C_rev<ValueType, tileSize>> Ctiles,
    r_Ptr<int> _C_perTileNnz,
    cr_Ptr<int> C_targetTiles,

    cr_Ptr<TileCSR_rev<ValueType,tileSize>> Atiles,
    cr_Ptr<int> _A_perTileNnz,

    cr_Ptr<TileCSR_rev<ValueType,tileSize>> Btiles,
    cr_Ptr<int> _B_perTileNnz
)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    extern __shared__ __align__(16) unsigned char tiles[];
    __shared__ long long perWarp_buffer[4 * 32];

    perWarp_buffer[threadIdx.x] = 0;
    
    // optimize later: streaming cache hint
    auto warp_load_tile_lambda = [](auto const &halfwarp, auto *tile, auto *halfwarp_tiles_start) __attribute__((always_inline)) 
    {
        int local_rank = halfwarp.thread_rank();
        if(local_rank == 0 || local_rank == 1)
        *(reinterpret_cast<ulonglong4*>(halfwarp_tiles_start) + local_rank) = *(reinterpret_cast<ulonglong4 const*>(tile) + local_rank);
    };

    int warp_pairs_idx_start = grid.block_rank() * 4 + warp.meta_group_rank();
    TileCSR_rev<ValueType> *warp_shmem_tile_start = reinterpret_cast<TileCSR_rev<ValueType>*>(tiles) + warp.meta_group_rank() * 2;
    while(warp_pairs_idx_start < d_pairs_size) 
    {
        auto halfwarp = cg::tiled_partition<16>(warp);
        int halfwarp_mgr = halfwarp.meta_group_rank();
        auto halfwarp_tile = (halfwarp_mgr == 0) ? Atiles : Btiles;
        // int halfwarp_tile_idx = ((d_pairs[warp_pairs_idx_start] >> (32 * (halfwarp_mgr ^ 0x1))) & 0xFFFF);
        int halfwarp_tile_idx = *(reinterpret_cast<int const*>(&d_pairs[warp_pairs_idx_start]) + (halfwarp_mgr ^ 0x1));

        int tile_nnz = (halfwarp_mgr == 0) 
                     ? (_A_perTileNnz[halfwarp_tile_idx+1] - _A_perTileNnz[halfwarp_tile_idx]) 
                     : (_B_perTileNnz[halfwarp_tile_idx+1] - _B_perTileNnz[halfwarp_tile_idx]);
        
        warp_load_tile_lambda(halfwarp, halfwarp_tile + halfwarp_tile_idx, warp_shmem_tile_start + halfwarp_mgr);
        warp.sync();

        // cg::memcpy_async(warp, warp_shmem_tile_start, Atiles + warp_Atile_idx, sizeof(TileCSR<ValueType>));
        // cg::memcpy_async(warp, warp_shmem_tile_start + 1, Btiles + _B_tileOffsets[warp_Btile_idx], sizeof(TileCSR<ValueType>));
        // cg::wait(warp);

        int warp_tileC_idx = C_targetTiles[warp_pairs_idx_start];
        TileCSR_C_rev<ValueType> *tileC = Ctiles + warp_tileC_idx;
        int tileC_nnz = _C_perTileNnz[warp_tileC_idx + 1] - _C_perTileNnz[warp_tileC_idx];

        __multiply_default_sync(warp, perWarp_buffer + tileSize * 2 * warp.meta_group_rank(), tileC, tileC_nnz, warp_shmem_tile_start + halfwarp_mgr, tile_nnz);

        warp_pairs_idx_start += grid.num_blocks() * block.num_threads() / warp.size();
    }
}

__device__ __forceinline__
void warp_exclusive_scan_sync(auto &warp_group, r_Ptr<int> arr, r_Ptr<int> temp_buffer){
    int my_val = arr[threadIdx.x];
    int ex_val = cg::exclusive_scan(warp_group, my_val);

    if(warp_group.thread_rank() == warp_group.size() - 1) {
        temp_buffer[warp_group.meta_group_rank()] = ex_val + my_val;
    }
    __syncthreads();

    if(warp_group.meta_group_rank() == 0) {
        int my_val2 = temp_buffer[warp_group.thread_rank()];
        int ex_val2 = cg::exclusive_scan(warp_group, my_val2);
        temp_buffer[warp_group.thread_rank()] = ex_val2;
    }
    __syncthreads();

    if(warp_group.meta_group_rank() > 0) {
        ex_val += temp_buffer[warp_group.meta_group_rank()];
    }
    __syncthreads();
    
    arr[threadIdx.x] = ex_val;
    __syncthreads();
}

template<typename ValueType, int tileSize = 16>
__global__ void __launch_bounds__(tileSize * tileSize)
sanitize_C
(
    r_Ptr<int> rows, 
    r_Ptr<int> cols, 
    r_Ptr<ValueType> vals, 
    cr_Ptr<TileCSR_C_rev<ValueType>> Ctiles, 
    cr_Ptr<int> _C_rowPtr, 
    int _C_rowPtrSize,
    cr_Ptr<int> _C_tileColIdx, 
    cr_Ptr<int> _C_perTile_Nnz
)
{
    __shared__ int permute[256];
    __shared__ int temp_buffer[8];
    __shared__ int this_tile_y;
    __shared__ int this_tile_x;

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int block_Ctiles_id = grid.block_rank();

    int block_offset = _C_perTile_Nnz[block_Ctiles_id];
    ValueType t_val = Ctiles[block_Ctiles_id].vals[block.thread_rank()];

    // maybe remove later
    permute [threadIdx.x] = 0;
    if(threadIdx.x < 8) temp_buffer[threadIdx.x] = 0;
    if(threadIdx.x == 0) this_tile_y = this_tile_x = 0;
    
    if(block.thread_rank() == 0) this_tile_y = lowerBound(_C_rowPtr, block_Ctiles_id, _C_rowPtrSize - 1);
    else if(block.thread_rank() == 1) this_tile_x = _C_tileColIdx[block_Ctiles_id];
    block.sync();

    if(block.thread_rank == 0) {
        printf("this_tile_y=%d this_tile_x=%d\n", this_tile_y, this_tile_x);
    }

    if(t_val != 0) permute[block.thread_rank()] = 1;
    block.sync();
    warp_exclusive_scan_sync(warp, permute, temp_buffer);

    if(t_val != 0) {
        rows[block_offset + permute[block.thread_rank()]] = this_tile_y * 16 + block.thread_rank() / tileSize;
        cols[block_offset + permute[block.thread_rank()]] = this_tile_x * 16 + block.thread_rank() % tileSize;
        vals[block_offset + permute[block.thread_rank()]] = t_val;
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

// #define USE_COOPERATIVE_LAUNCH
// #define DEBUG_1
// #define DEBUG_2
// #define DEBUG_3
// #define DEBUG_4
// #define DEBUG_5
// #define DEBUG_6
// #define DEBUG_7
// #define DEBUG_8
// #define DEBUG_9
// #define DEBUG_10
// #define DEBUG_11

int main(int argc, char *argv[]) {
    if(argc <= 1 || argc > 3) {
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
    argv[2] = argv[1]; // WARNING DELETE LATER

    std::jthread read_A(read_matrix_market<ValueType>, std::ref(argv[1]), std::ref(A_I), std::ref(A_J), std::ref(A_val), std::ref(A_rows), std::ref(A_cols), std::ref(A_nnz));
    // read_matrix_market(argv[1], A_I, A_J, A_val, A_rows, A_cols, A_nnz);
    read_matrix_market(argv[2], B_I, B_J, B_val, B_rows, B_cols, B_nnz);
    read_A.join();

    std::cout 
    << "MATRIX A\n"
    << "filepath: " << argv[1] << "\n"
    << "Rows: " << A_rows << "\n"
    << "Cols: " << A_cols << "\n"
    << "Nnz: " << A_nnz << "\n";

    std::cout 
    << "MATRIX B\n"
    << "filepath: " << argv[2] << "\n"
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

    thrust::copy(ASYNC_EXEC_POLICY(STREAM_B), B_I.begin(), B_I.end(), B_d_I.begin());
    thrust::copy(ASYNC_EXEC_POLICY(STREAM_B), B_J.begin(), B_J.end(), B_d_J.begin());
    thrust::copy(ASYNC_EXEC_POLICY(STREAM_B), B_val.begin(), B_val.end(), B_d_val.begin());

    if(argv[2] == "TRANSPOSE"){
        auto zit = thrust::make_zip_iterator(thrust::make_tuple(B_d_J.begin(), B_d_I.begin(), B_val.begin()));
        thrust::sort(ASYNC_EXEC_POLICY(STREAM_B), zit, zit + B_nnz);
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
        A_tileRows, 
        A_nnz
    );
    dim3 B_threads_dwc{tileSize * tileSize};
    dim3 B_blocks_dwc{(B_nnz - 1 + B_threads_dwc.x)/B_threads_dwc.x};
    decide_which_tile<<<B_blocks_dwc, B_threads_dwc, 0, STREAM_B>>>
    (
        B_participating_tiles.data().get(), 
        B_d_I.data().get(), 
        B_d_J.data().get(), 
        B_tileCols, 
        B_nnz
    );
    
    rmm::device_vector<int> A_perTileNnz(SPGEMM_STREAM_ALLOCATOR_INT(STREAM_A));
    {
    thrust::sort(ASYNC_EXEC_POLICY(STREAM_A), A_participating_tiles.begin(), A_participating_tiles.end(), thrust::less<long long>());
    
    int cnt = thrust::unique_count(ASYNC_EXEC_POLICY(STREAM_A), A_participating_tiles.begin(), A_participating_tiles.end());
    A_perTileNnz.resize(cnt + 1);
    thrust::reduce_by_key(ASYNC_EXEC_POLICY(STREAM_A), A_participating_tiles.begin(), A_participating_tiles.end(), thrust::make_constant_iterator<int>(1), thrust::make_discard_iterator(), A_perTileNnz.begin());
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_A),A_perTileNnz.begin(), A_perTileNnz.end(), A_perTileNnz.begin());

    auto newend = thrust::unique(ASYNC_EXEC_POLICY(STREAM_A), A_participating_tiles.begin(), A_participating_tiles.end());
    A_participating_tiles.erase(newend, A_participating_tiles.end());
    }

    rmm::device_vector<int> B_perTileNnz(SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B));
    {
    thrust::sort(ASYNC_EXEC_POLICY(STREAM_B), B_participating_tiles.begin(), B_participating_tiles.end(), thrust::less<long long>());
    
    int cnt = thrust::unique_count(ASYNC_EXEC_POLICY(STREAM_B), B_participating_tiles.begin(), B_participating_tiles.end());
    B_perTileNnz.resize(cnt + 1);
    thrust::reduce_by_key(ASYNC_EXEC_POLICY(STREAM_B), B_participating_tiles.begin(), B_participating_tiles.end(), thrust::make_constant_iterator<int>(1), thrust::make_discard_iterator(), B_perTileNnz.begin());
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_B), B_perTileNnz.begin(), B_perTileNnz.end(), B_perTileNnz.begin());

    auto newend = thrust::unique(ASYNC_EXEC_POLICY(STREAM_B), B_participating_tiles.begin(), B_participating_tiles.end());
    B_participating_tiles.erase(newend, B_participating_tiles.end());
    }

#ifdef DEBUG_1
    std::jthread DEBUG_1([&]()
    {
        {
        char constexpr *filename = "../src/DEBUG/DEBUG_A_1";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);
        thrustHvec<long long> h_participating_tiles(A_participating_tiles.size());
        
        cudaStreamSynchronize(STREAM_A); 
        auto hA_perTileNnz = A_perTileNnz;
        outfile << "A_perTileNnz\n";
        for(int i = 0; i < hA_perTileNnz.size(); ++i) {
            outfile << hA_perTileNnz[i] << " ";
            if((i+1)%64 == 0) outfile << "\n";
        }
        thrust::copy(A_participating_tiles.begin(), A_participating_tiles.end(), h_participating_tiles.begin());
        outfile << "size: " << h_participating_tiles.size() << "\n";
        outfile << "participating tiles: [";
        for(size_t i = 0; i < h_participating_tiles.size(); ++i) {
            int x = *(reinterpret_cast<int*>(&h_participating_tiles[i]));
            int y = *(reinterpret_cast<int*>(&h_participating_tiles[i])+1);
            outfile << "i: " << i << " (" << y << ", " << x << "), ";
        }
        outfile << "]" << std::endl;
        outfile << "len: " << A_participating_tiles.size() << std::endl;

        outfile.close();
        }

        {
        char constexpr *filename = "../src/DEBUG/DEBUG_B_1";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);
        thrustHvec<long long> h_participating_tiles(B_participating_tiles.size());
        
        cudaStreamSynchronize(STREAM_B); 
        auto hB_perTileNnz = B_perTileNnz;
        outfile << "B_perTileNnz\n";
        for(int i = 0; i < hB_perTileNnz.size(); ++i) {
            outfile << hB_perTileNnz[i] << " ";
            if((i+1)%64 == 0) outfile << "\n";
        }
        thrust::copy(B_participating_tiles.begin(), B_participating_tiles.end(), B_participating_tiles.begin());
        outfile << "size: " << h_participating_tiles.size() << "\n";
        outfile << "participating tiles: [";
        for(size_t i = 0; i < h_participating_tiles.size(); ++i) {
            int x = *(reinterpret_cast<int*>(&h_participating_tiles[i]));
            int y = *(reinterpret_cast<int*>(&h_participating_tiles[i])+1);
            outfile << "i: " << i << " (" << y << ", " << x << "), ";
        }
        outfile << "]" << std::endl;
        outfile << "len: " << B_participating_tiles.size() << std::endl;
        outfile.close();
        }
    });
#endif

    rmm::device_vector<int> A_d_rowPtr(A_rows + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_A));
    {
    auto zit = thrust::make_zip_iterator(thrust::make_tuple(A_d_I.begin(), A_d_J.begin(), A_d_val.begin()));
    thrust::stable_sort(ASYNC_EXEC_POLICY(STREAM_A), zit, zit + A_nnz);
    thrust::reduce_by_key(ASYNC_EXEC_POLICY(STREAM_A), A_d_I.begin(), A_d_I.end(), thrust::make_constant_iterator<int>(1), thrust::make_discard_iterator(), A_d_rowPtr.begin());
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_A), A_d_rowPtr.begin(), A_d_rowPtr.end(), A_d_rowPtr.begin());
    }

    rmm::device_vector<int> B_d_rowPtr(B_rows + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B));
    {
    auto zit = thrust::make_zip_iterator(thrust::make_tuple(B_d_I.begin(), B_d_J.begin(), B_d_val.begin()));
    thrust::stable_sort(ASYNC_EXEC_POLICY(STREAM_B), zit, zit + B_nnz);
    thrust::reduce_by_key(ASYNC_EXEC_POLICY(STREAM_B), B_d_I.begin(), B_d_I.end(), thrust::make_constant_iterator<int>(1), thrust::make_discard_iterator(), B_d_rowPtr.begin());
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_B), B_d_rowPtr.begin(), B_d_rowPtr.end(), B_d_rowPtr.begin());
    }

#ifdef DEBUG_2
    std::jthread DEBUG_2([&](){
        {
        char const *filename = "../src/DEBUG/DEBUG_A_2";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        cudaStreamSynchronize(STREAM_A); 
        outfile << "d_I: [";
        thrust::copy(A_d_I.begin(), A_d_I.end(), std::ostream_iterator<int>(outfile, ", "));
        outfile << "]" << std::endl;        
        outfile << "d_J: [";
        thrust::copy(A_d_J.begin(), A_d_J.end(), std::ostream_iterator<int>(outfile, ", "));
        outfile << "]" << std::endl;
        outfile << "d_rowPtr: [";
        thrust::copy(A_d_rowPtr.begin(), A_d_rowPtr.end(), std::ostream_iterator<int>(outfile, ", "));
        outfile << "]" << std::endl;

        outfile.close();
        }
        {
        char const *filename = "../src/DEBUG/DEBUG_B_2";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        cudaStreamSynchronize(STREAM_B); 
        outfile << "d_I: [";
        thrust::copy(B_d_I.begin(), B_d_I.end(), std::ostream_iterator<int>(outfile, ", "));
        outfile << "]" << std::endl;        
        outfile << "d_J: [";
        thrust::copy(B_d_J.begin(), B_d_J.end(), std::ostream_iterator<int>(outfile, ", "));
        outfile << "]" << std::endl;
        outfile << "d_rowPtr: [";
        thrust::copy(B_d_rowPtr.begin(), B_d_rowPtr.end(), std::ostream_iterator<int>(outfile, ", "));
        outfile << "]" << std::endl;

        outfile.close();
        }
    });
#endif

    dim3 A_threads_gtc {tileSize * tileSize};
    dim3 A_blocks_gtc {A_participating_tiles.size()};

    rmm::device_vector<TileCSR_rev<ValueType>> Atiles(A_participating_tiles.size(), SPGEMM_STREAM_ALLOCATOR_TILECSR_REV(STREAM_A));
    rmm::device_vector<ValueType> Atiles_vals(A_nnz, SPGEMM_STREAM_ALLOCATOR_VALUETYPE(STREAM_A));
    rmm::device_vector<uint8_t> Atiles_rowColIdx(A_nnz, SPGEMM_STREAM_ALLOCATOR_UINT8(STREAM_A));
    rmm::device_vector<int> 
    A_d_cols(1, A_cols, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_A)), 
    A_participating_tiles_size(1, A_participating_tiles.size(), SPGEMM_STREAM_ALLOCATOR_INT(STREAM_A));

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

    dim3 B_threads_gtc {tileSize * tileSize};
    dim3 B_blocks_gtc {B_participating_tiles.size()};

    rmm::device_vector<TileCSR_rev<ValueType>> Btiles(B_participating_tiles.size(), SPGEMM_STREAM_ALLOCATOR_TILECSR_REV(STREAM_B));
    rmm::device_vector<ValueType> Btiles_vals(B_nnz, SPGEMM_STREAM_ALLOCATOR_VALUETYPE(STREAM_B));
    rmm::device_vector<uint8_t> Btiles_rowColIdx(B_nnz, SPGEMM_STREAM_ALLOCATOR_UINT8(STREAM_B));
    rmm::device_vector<int> 
    B_d_cols(1, B_cols, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B)), 
    B_participating_tiles_size(1, B_participating_tiles.size(), SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B));

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


#ifdef USE_COOPERATIVE_LAUNCH

    constexpr size_t max_tb_RTX3080 = (1536 / (tileSize * tileSize)) * 48;
    int maxBlocksPerMultiprocessor;
    int multiProcessorCount;
    int sharedMemoryPerBlock = 256; // Replace with actual value
    int maxSharedMemoryPerGPU = 49152; // Replace with actual value (in bytes)
    int registersPerBlock = 32; // Replace with actual value
    int maxRegistersPerGPU = 65536; // Replace with actual value

    cudaDeviceGetAttribute(&maxBlocksPerMultiprocessor, cudaDevAttrMaxBlocksPerMultiprocessor, 0);
    cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, 0);

    int maxBlocksPerGPU = maxBlocksPerMultiprocessor * multiProcessorCount;
    int blockSizeLimitSharedMemory = maxSharedMemoryPerGPU / sharedMemoryPerBlock;
    int blockSizeLimitRegisters = maxRegistersPerGPU / registersPerBlock;

    int blockSizeLimit = min(blockSizeLimitSharedMemory, blockSizeLimitRegisters);
    int maxBlocks = min(maxBlocksPerGPU, blockSizeLimit);

    printf("Maximum allowable number of blocks: %d\n", maxBlocks);
    blocks_gtc.x = maxBlocks;

    auto Atiles_ptr = Atiles.data().get();
    auto participating_tiles_ptr = participating_tiles.data().get();
    auto _participating_tiles_size_ptr = _participating_tiles_size.data().get();
    auto AtilePtr_ptr = AtilePtr.data().get();
    auto AtileColIdx_ptr = AtileColIdx.data().get();
    auto AtileNnz_ptr = AtileNnz.data().get();
    auto d_J_ptr = d_J.data().get();
    auto d_val_ptr = d_val.data().get();
    auto d_rowPtr_ptr = d_rowPtr.data().get();
    auto d_rowPtr_size_ptr = _d_rowPtr_size.data().get();
    auto _cols_ptr = _cols.data().get();
    auto _nnz_ptr = _nnz.data().get();

    void *kernelArgs[] = {
        static_cast<void*>(&Atiles_ptr),
        static_cast<void*>(&participating_tiles_ptr),
        static_cast<void*>(&_participating_tiles_size_ptr),
        static_cast<void*>(&AtilePtr_ptr),
        static_cast<void*>(&AtileColIdx_ptr),
        static_cast<void*>(&AtileNnz_ptr),
        static_cast<void*>(&d_J_ptr),
        static_cast<void*>(&d_val_ptr),
        static_cast<void*>(&d_rowPtr_ptr),
        static_cast<void*>(&d_rowPtr_size_ptr),
        static_cast<void*>(&_cols_ptr),
        static_cast<void*>(&_nnz_ptr),
    };

    CHECK_CUDA(cudaLaunchCooperativeKernel((void*)generate_tiles_csr<int, tileSize>, blocks_gtc, threads_gtc, kernelArgs, 0, streams[STREAM_A]));
#endif

#ifdef DEBUG_3
    std::jthread DEBUG_3([&](){
        {
        char constexpr *filename = "../src/DEBUG/DEBUG_A_3";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        thrustHvec<decltype(Atiles[0])::value_type> hAtiles(A_participating_tiles.size());
        thrustHvec<long long> phAtiles(A_participating_tiles.size());

        cudaStreamSynchronize(STREAM_A);
        thrustHvec<ValueType> hAtiles_vals = Atiles_vals;
        thrustHvec<uint8_t> hAtiles_rowColIdx = Atiles_rowColIdx;
        outfile << "]\n" << std::endl;
        hAtiles = Atiles;
        phAtiles = A_participating_tiles;
        outfile << "Atiles: [(y, x)" << std::endl;
        for(int i = 0; i < hAtiles.size(); ++i) {
            int x = *(reinterpret_cast<int*>(&phAtiles[i]));
            int y = *(reinterpret_cast<int*>(&phAtiles[i])+1);
            outfile << "Tile" << "(" << y << ", " << x << ") ";
            hAtiles[i].vals = hAtiles_vals.data() + A_perTileNnz[i];
            hAtiles[i].rowColIdx = hAtiles_rowColIdx.data() + A_perTileNnz[i];
            printInfo(outfile, hAtiles[i], A_perTileNnz[i+1]-A_perTileNnz[i]);
        }
        outfile << "]" << std::endl;
        outfile << "TileNnz: [";
        for(int i = 0; i < A_perTileNnz.size(); ++i) {
            outfile << A_perTileNnz[i] << " ";
        }
        outfile << "]" << std::endl;

        outfile.close();
        }

        {
        char constexpr *filename = "../src/DEBUG/DEBUG_B_3";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        thrustHvec<decltype(Btiles[0])::value_type> hBtiles(B_participating_tiles.size());
        thrustHvec<long long> phBtiles(B_participating_tiles.size());

        cudaStreamSynchronize(STREAM_B);
        thrustHvec<ValueType> hBtiles_vals = Btiles_vals;
        thrustHvec<uint8_t> hBtiles_rowColIdx = Btiles_rowColIdx;
        outfile << "]\n" << std::endl;
        hBtiles = Btiles;
        phBtiles = B_participating_tiles;
        outfile << "Btiles: [(y, x)" << std::endl;
        for(int i = 0; i < hBtiles.size(); ++i) {
            int x = *(reinterpret_cast<int*>(&phBtiles[i]));
            int y = *(reinterpret_cast<int*>(&phBtiles[i])+1);
            outfile << "Tile" << "(" << y << ", " << x << ") ";
            hBtiles[i].vals = hBtiles_vals.data() + B_perTileNnz[i];
            hBtiles[i].rowColIdx = hBtiles_rowColIdx.data() + B_perTileNnz[i];
            printInfo(outfile, hBtiles[i], B_perTileNnz[i+1]-B_perTileNnz[i]);
        }
        outfile << "]" << std::endl;
        outfile << "TileNnz: [";
        for(int i = 0; i < B_perTileNnz.size(); ++i) {
            outfile << B_perTileNnz[i] << " ";
        }
        outfile << "]" << std::endl;

        outfile.close();
        }
    });
#endif

    // create High level Representation of A -> A_
    rmm::device_vector<int> _A_tileRowPtr(A_tileRows + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_D));
    rmm::device_vector<int> _A_tileColIdx(A_participating_tiles.size(), SPGEMM_STREAM_ALLOCATOR_INT(STREAM_D));
    rmm::device_vector<float> _A_tileVals(A_participating_tiles.size(), SPGEMM_STREAM_ALLOCATOR_FLOAT(STREAM_D));
    
    thrust::reduce_by_key(
        ASYNC_EXEC_POLICY(STREAM_D),
        thrust::make_transform_iterator(A_participating_tiles.begin(), getHigh32()),
        thrust::make_transform_iterator(A_participating_tiles.end(), getHigh32()),
        thrust::make_constant_iterator<int>(1),
        thrust::make_discard_iterator(),
        _A_tileRowPtr.begin()
    );
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_D), 
    _A_tileRowPtr.begin(), _A_tileRowPtr.end(), _A_tileRowPtr.begin());

    thrust::copy(
        ASYNC_EXEC_POLICY(STREAM_D),
        thrust::make_transform_iterator(A_participating_tiles.begin(), getLow32()),
        thrust::make_transform_iterator(A_participating_tiles.end(), getLow32()),
        _A_tileColIdx.begin()
    );

    // create High level Representation of B -> B_
    rmm::device_vector<int> _B_tileRowPtr(B_tileRows + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    rmm::device_vector<int> _B_tileColIdx(B_participating_tiles.size(), SPGEMM_STREAM_ALLOCATOR_INT(STREAM_E));
    rmm::device_vector<float> _B_tileVals(B_participating_tiles.size(), SPGEMM_STREAM_ALLOCATOR_FLOAT(STREAM_E));
    
    thrust::reduce_by_key(
        ASYNC_EXEC_POLICY(STREAM_E),
        thrust::make_transform_iterator(B_participating_tiles.begin(), getHigh32()),
        thrust::make_transform_iterator(B_participating_tiles.end(), getHigh32()),
        thrust::make_constant_iterator<int>(1),
        thrust::make_discard_iterator(),
        _B_tileRowPtr.begin()
    );
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_E), 
    _B_tileRowPtr.begin(), _B_tileRowPtr.end(), _B_tileRowPtr.begin());

    thrust::copy(
        ASYNC_EXEC_POLICY(STREAM_E),
        thrust::make_transform_iterator(B_participating_tiles.begin(), getLow32()),
        thrust::make_transform_iterator(B_participating_tiles.end(), getLow32()),
        _B_tileColIdx.begin()
    );

    std::jthread destroy_participating_tiles([&A_participating_tiles, &B_participating_tiles, &streams](){
        cudaStreamSynchronize(STREAM_D);
        rmm::device_vector<long long>().swap(A_participating_tiles);
        cudaStreamSynchronize(STREAM_E);
        rmm::device_vector<long long>().swap(B_participating_tiles);
    });

#ifdef DEBUG_4
    std::jthread DEBUG_4([&](){
        {
        char const *filename = "../src/DEBUG/DEBUG_A_4";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        thrustHvec<int> hAtilePtr = _A_tileRowPtr;
        thrustHvec<int> hAtileColIdx = _A_tileColIdx;
        thrustHvec<float> hAtileVal = _A_tileVals;

        cudaStreamSynchronize(STREAM_A);

        outfile << "AtileRowPtr: [\n";
        for(int i = 0; i < hAtilePtr.size(); ++i) {
            outfile << hAtilePtr.data()[i] << " ";
        }
        outfile << "\n]" << std::endl;
        outfile << "\nAtileColIdx: [\n";
        for(int i = 0; i < hAtileColIdx.size(); ++i) {
            outfile << hAtileColIdx.data()[i] << " ";
        }
        outfile << "\n]" << std::endl;
        outfile << "AtileVal: [";
        for(int i = 0; i < hAtileVal.size(); ++i) {
            outfile << hAtileVal.data()[i] << " ";
        }
        outfile << "]" << std::endl;

        outfile.close();
        }
        {
        char const *filename = "../src/DEBUG/DEBUG_B_4";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        thrustHvec<int> hBtilePtr = _B_tileRowPtr;
        thrustHvec<int> hBtileColIdx = _B_tileColIdx;
        thrustHvec<float> hBtileVal = _B_tileVals;

        cudaStreamSynchronize(STREAM_B);

        outfile << "BtileRowPtr: [";
        for(int i = 0; i < hBtilePtr.size(); ++i) {
            outfile << hBtilePtr.data()[i] << " ";
        }
        outfile << "]" << std::endl;
        outfile << "BtileColIdx: [";
        for(int i = 0; i < hBtileColIdx.size(); ++i) {
            outfile << hBtileColIdx.data()[i] << " ";
        }
        outfile << "]" << std::endl;
        outfile << "BtileVal: [";
        for(int i = 0; i < hBtileVal.size(); ++i) {
            outfile << hBtileVal.data()[i] << " ";
        }
        outfile << "]" << std::endl;

        outfile.close();
        }
    });
#endif
    
    rmm::device_vector<int> _C_tilePtr(SPGEMM_STREAM_ALLOCATOR_INT(STREAM_C)), _C_tileColIdx(SPGEMM_STREAM_ALLOCATOR_INT(STREAM_C));
    cudaStreamSynchronize(STREAM_D);
    cudaStreamSynchronize(STREAM_E);
    cusparse_highLevelMultiply
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
        &_C_tilePtr, &_C_tileColIdx,
        STREAM_C,
        ASYNC_EXEC_POLICY(STREAM_C)
    );

#ifdef DEBUG_5
    std::jthread DEBUG_5([&](){
        char const *filename = "../src/DEBUG/DEBUG_C_5";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        cudaStreamSynchronize(STREAM_C);
        thrustHvec<int> hCtilePtr = _C_tilePtr;
        thrustHvec<int> hCtileColIdx = _C_tileColIdx;
        outfile << "hCtilePtr nums: " << hCtilePtr.size() << "\n";
        outfile << "hCtileColIdx nums: " << hCtileColIdx.size() << "\n";
        outfile << "_C_tilePtr: [\n";
        for(int i = 0; i < hCtilePtr.size(); ++i) {
            outfile << hCtilePtr.data()[i] << " ";
        }
        outfile << "\n]" << std::endl;
        outfile << "\n_C_tileColIdx: [\n";
        for(int i = 0; i < hCtileColIdx.size(); ++i) {
            outfile << hCtileColIdx.data()[i] << " ";
        }
        outfile << "\n]" << std::endl;

        outfile.close();
    });
#endif

    // transpose _B
    dim3 threads_tr {tileSize * tileSize};
    dim3 blocks_tr {(_B_tileRowPtr.size()-1 - 1 + threads_tr.x)/threads_tr.x};
    rmm::device_vector<int> _B_tileRowIdx(_B_tileColIdx.size(), SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B));
    _tag_rows<<<blocks_tr, threads_tr, 0, STREAM_B>>>(_B_tileRowIdx.data().get(), _B_tileRowPtr.data().get(), _B_tileRowPtr.size());
    thrust::inclusive_scan(ASYNC_EXEC_POLICY(STREAM_B), _B_tileRowIdx.begin(), _B_tileRowIdx.end(), _B_tileRowIdx.begin());
    rmm::device_vector<int> _B_tileColPtr(B_tileCols + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B));
    rmm::device_vector<int> _B_tileOffsets(_B_tileColIdx.size(), SPGEMM_STREAM_ALLOCATOR_INT(STREAM_B));
    {
        thrust::sequence(ASYNC_EXEC_POLICY(STREAM_B), _B_tileOffsets.begin(), _B_tileOffsets.end());

        auto zit = thrust::make_zip_iterator(thrust::make_tuple(_B_tileColIdx.begin(), _B_tileRowIdx.begin(), _B_tileOffsets.begin()));
        thrust::stable_sort(ASYNC_EXEC_POLICY(STREAM_B), zit, zit + _B_tileColIdx.size());
        thrust::reduce_by_key(ASYNC_EXEC_POLICY(STREAM_B), _B_tileColIdx.begin(), _B_tileColIdx.end(), thrust::make_constant_iterator<int>(1), thrust::make_discard_iterator(), _B_tileColPtr.begin());
        thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_B), _B_tileColPtr.begin(), _B_tileColPtr.end(), _B_tileColPtr.begin());

        _B_tileRowPtr.clear();
        rmm::device_vector<int>().swap(_B_tileRowPtr);
    }

#ifdef DEBUG_6
    std::jthread DEBUG_6([&](){
        char const *filename = "../src/DEBUG/DEBUG_B_6";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        cudaStreamSynchronize(STREAM_B);
        outfile << "_B_tileColPtr: [\n";
        for(int i = 0; i < _B_tileColPtr.size(); ++i) {
            outfile << _B_tileColPtr.data()[i] << " ";
        }
        outfile << "\n]" << std::endl;
        outfile << "\n_B_tileRowIdx: [\n";
        for(int i = 0; i < _B_tileRowIdx.size(); ++i) {
            outfile << _B_tileRowIdx.data()[i] << " ";
        }
        outfile << "\n]" << std::endl;

        outfile.close();
    });
#endif

    std::cout << "\nSPECULATING PAIRS\n";

    int numBlocksPerSm_sp = 0;
    int numThreads_sp = 256;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm_sp, search_pairs<0>, numThreads_sp, 0);
    std::cout << "search_pairs<0> Max block count per SM with 256 threads: " << numBlocksPerSm_sp << "\n";

    dim3 threads_sp {numThreads_sp};
    dim3 blocks_sp {numBlocksPerSm_sp * deviceProp.multiProcessorCount};

    int *_C_tilePtr_size, *_C_tileColIdx_size;
    int CtilePtrSize = _C_tilePtr.size();
    int CtileColIdxSize = _C_tileColIdx.size();
    cudaMallocAsync(&_C_tilePtr_size, sizeof(int), STREAM_C);
    cudaMallocAsync(&_C_tileColIdx_size, sizeof(int), STREAM_C);
    cudaMemcpyAsync(_C_tilePtr_size, &CtilePtrSize, sizeof(int), cudaMemcpyHostToDevice, STREAM_C);
    cudaMemcpyAsync(_C_tileColIdx_size, &CtileColIdxSize, sizeof(int), cudaMemcpyHostToDevice, STREAM_C);

    int *pairs_count;
    cudaMallocManaged(&pairs_count, sizeof(int));
    cudaMemsetAsync(pairs_count, 0, sizeof(int), STREAM_C);

    // auto ptr1 = d_pairs.data().get();
    long long *ptr1 = nullptr;
    int *ptr2 = nullptr;
    auto ptr3 = _C_tilePtr.data().get();
    auto ptr4 = _C_tileColIdx.data().get();
    auto ptr5 = _A_tileRowPtr.data().get();
    auto ptr6 = _A_tileColIdx.data().get();
    auto ptr7 = _B_tileColPtr.data().get();
    auto ptr8 = _B_tileRowIdx.data().get();
    auto ptr9 = _C_tilePtr_size;
    auto ptr10 = _C_tileColIdx_size;
    auto ptr11 = _B_tileOffsets.data().get();

    void *kern_args[] = {
        static_cast<void*>(&ptr1),
        static_cast<void*>(&ptr2),
        static_cast<void*>(&ptr3),
        static_cast<void*>(&ptr4),
        static_cast<void*>(&ptr5),
        static_cast<void*>(&ptr6),
        static_cast<void*>(&ptr7),
        static_cast<void*>(&ptr8),
        static_cast<void*>(&ptr9),
        static_cast<void*>(&ptr10),
        static_cast<void*>(&ptr11),
        static_cast<void*>(&pairs_count)
    };

    cudaStreamSynchronize(STREAM_B); // _B_tileOffsets comes from STREAM_B
    // first pass
    CHECK_CUDA( cudaLaunchCooperativeKernel((void*)search_pairs<0>, blocks_sp, threads_sp, kern_args, 0, STREAM_C) )
    cudaStreamSynchronize(STREAM_C);
    rmm::device_vector<long long> d_pairs(*pairs_count, SPGEMM_STREAM_ALLOCATOR_LONGLONG(STREAM_C));
    rmm::device_vector<int> C_targetTile(*pairs_count, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_C));
    std::cout << "FIRST PASS pairs count: " << *pairs_count << "\n";
    // second pass
    ptr1 = d_pairs.data().get();
    ptr2 = C_targetTile.data().get();
    kern_args[0] = static_cast<void*>(&ptr1);
    kern_args[1] = static_cast<void*>(&ptr2);
    cudaMemsetAsync(pairs_count, 0, sizeof(int), STREAM_C);
    CHECK_CUDA( cudaLaunchCooperativeKernel((void*)search_pairs<1>, blocks_sp, threads_sp, kern_args, 0, STREAM_C) )

#ifdef DEBUG_9
    std::jthread debug_intersection([&]()
    {
        cudaStreamSynchronize(STREAM_C);
        std::cout << "SECOND PASS pairs count: " << *pairs_count << "\n";
        thrustDvec<long long> h_pairs = d_pairs;
        thrust::sort(h_pairs.begin(), h_pairs.end(), thrust::less<long long>());
        thrust::sort(d_pairs.begin(), d_pairs.end(), thrust::less<long long>());
        auto mismatch = thrust::mismatch(d_pairs.begin(), d_pairs.end(), h_pairs.begin());
        if(mismatch.first == d_pairs.end()) std::cout << "d_pairs = h_pairs\n";
        else {
        std::cout << "d_pairs != h_pairs at ";
        std::cout << mismatch.first - d_pairs.begin() << "\n";
        std::cout << *mismatch.first << " x " << *mismatch.second << "\n";
        }
        {
        char const *filename = "../src/DEBUG/DEBUG_C_9";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        auto print = [&outfile] (auto v) { outfile << v << "\n"; };
        
        thrust::sort(C_targetTile.begin(), C_targetTile.end());
        thrustHvecPin<int> hCtargetTiles = C_targetTile;

        outfile << "MATRIX C targetTile -- RESULT\n";
        outfile << "nums: " << hCtargetTiles.size() << "\n";
        std::for_each(hCtargetTiles.begin(), hCtargetTiles.end(), print);
        outfile.close();
        }
        {
        char const *filename = "../src/DEBUG/DEBUG_C_9_2";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        auto print = [&outfile] (long long v) {
            int A = v >> 32;
            int B = v & 0xFFFFFFFF; 
            outfile << "A: " << A << " B: " << B << "\n"; 
        };

        outfile << "PAIRS -- RESULT\n";
        outfile << "nums: " << h_pairs.size() << "\n";
        std::for_each(h_pairs.begin(), h_pairs.end(), print);
        outfile.close();
        }
    });
#endif

    rmm::device_vector<TileCSR_C_rev<ValueType>> Ctiles(_C_tileColIdx.size(), SPGEMM_STREAM_ALLOCATOR_TILECSRC_REV(STREAM_C));
    rmm::device_vector<int> _C_perTileNnz(_C_tileColIdx.size() + 1, SPGEMM_STREAM_ALLOCATOR_INT(STREAM_C));
    
    int numBlocksPerSm_aC = 0;
    int numThreads_aC = 128;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm_aC, (void*)allocate_C<ValueType>, numThreads_aC, 0);
    std::cout << "allocate_C Max block count per SM with 128 threads: " << numBlocksPerSm_aC << "\n";

    auto args1 = d_pairs.data().get();
    auto args2 = d_pairs.size();
    auto args3 =  Ctiles.data().get();
    auto args4 = Ctiles.size();
    auto args5 = _C_perTileNnz.data().get();
    auto args6 = C_targetTile.data().get();
    auto args7 =  Atiles.data().get();
    auto args8 =  Btiles.data().get();

    void *args[] = {
        static_cast<void*>(&args1),
        static_cast<void*>(&args2),
        static_cast<void*>(&args3),
        static_cast<void*>(&args4),
        static_cast<void*>(&args5),
        static_cast<void*>(&args6),
        static_cast<void*>(&args7),
        static_cast<void*>(&args8),
    };

    dim3 threads_aC {numThreads_aC};
    dim3 blocks_aC {numBlocksPerSm_aC * deviceProp.multiProcessorCount};
    CHECK_CUDA( cudaLaunchCooperativeKernel((void*)allocate_C<ValueType>, blocks_aC, threads_aC, args, 0, STREAM_C) )
    thrust::exclusive_scan(ASYNC_EXEC_POLICY(STREAM_C), _C_perTileNnz.begin(), _C_perTileNnz.end(), _C_perTileNnz.begin());

    rmm::device_vector<ValueType> Ctiles_vals(_C_perTileNnz.back(), SPGEMM_STREAM_ALLOCATOR_VALUETYPE(STREAM_C));
    rmm::device_vector<typename TileCSR_C_rev<ValueType>::IdxType> Ctiles_rowColIdx(_C_perTileNnz.back(), SPGEMM_STREAM_ALLOCATOR_UINT8(STREAM_C));

    int numBlocksPerSm_Cs = 0;
    int numThreads_Cs = 128;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm_Cs, (void*)C_setOffsets<ValueType>, numThreads_Cs, 0);
    std::cout << "C_setOffsets Max block count per SM with 128 threads: " << numBlocksPerSm_Cs << "\n";

    auto args9 = Ctiles_vals.data().get();
    auto args10 = Ctiles_rowColIdx.data().get();
    void *args_2[] = {
        static_cast<void*>(&args3),
        static_cast<void*>(&args4),
        static_cast<void*>(&args5),
        static_cast<void*>(&args9),
        static_cast<void*>(&args10)
    };

    dim3 threads_Cs {numThreads_Cs};
    dim3 blocks_Cs {numBlocksPerSm_Cs * deviceProp.multiProcessorCount};
    CHECK_CUDA( cudaLaunchCooperativeKernel((void*)C_setOffsets<ValueType>, blocks_Cs, threads_Cs, args_2, 0, STREAM_C) ); // change blocks and threads later?

#ifdef DEBUG_10
    std::jthread DEBUG_10([&]{
        cudaStreamSynchronize(STREAM_C);
        char const *filename = "../src/DEBUG/DEBUG_C_10";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        outfile << "C nnz\n" << _C_perTileNnz.back() << "\n\n";

        thrustHvecPin<TileCSR_C_rev<ValueType>> hCtiles = Ctiles;
        thrustHvecPin<int> hCtilesNnz = _C_perTileNnz;
        for(int n = 0; n < hCtiles.size(); ++n)
        {
            outfile << "TILE " << n << "\n";
            printInfo(outfile, hCtiles[n], hCtilesNnz[n+1] - hCtilesNnz[n]);
            outfile << "\n";
        }
        outfile.close();
    });
#endif  

    auto arg1 = d_pairs.data().get();
    auto arg2 = d_pairs.size();
    auto arg3 =  Ctiles.data().get();
    auto arg4 =  _C_perTileNnz.data().get();
    auto arg5 = C_targetTile.data().get();
    auto arg6 =  Atiles.data().get();
    auto arg7 =  A_perTileNnz.data().get();
    auto arg8 =  Btiles.data().get();
    auto arg9 =  B_perTileNnz.data().get();

    void *kernArgs[] = {
        static_cast<void*>(&arg1),
        static_cast<void*>(&arg2),
        static_cast<void*>(&arg3),
        static_cast<void*>(&arg4),
        static_cast<void*>(&arg5),
        static_cast<void*>(&arg6),
        static_cast<void*>(&arg7),
        static_cast<void*>(&arg8),
        static_cast<void*>(&arg9),
    };

    int numBlocksPerSm_mp = 0;
    int numThreads_mp = 128;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm_mp, (void*)multiply_pairs<ValueType>, numThreads_mp, sizeof(TileCSR_rev<ValueType>) * 8);
    std::cout << "multiply_pairs Max block count per SM with 128 threads: " << numBlocksPerSm_mp << "\n";

    dim3 threads_mp {numThreads_mp};
    dim3 blocks_mp {numBlocksPerSm_mp * deviceProp.multiProcessorCount};
    CHECK_CUDA( cudaLaunchCooperativeKernel((void*)multiply_pairs<ValueType>, blocks_mp, threads_mp, kernArgs, sizeof(TileCSR_rev<ValueType>) * 8, STREAM_C) )

#ifdef DEBUG_11
    std::jthread DEBUG_11([&]{
        cudaStreamSynchronize(STREAM_C);
        std::cout << "ALGORITHM FINISHED\n";
        char const *filename = "../src/DEBUG/DEBUG_C_11";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        outfile << "C nnz\n" << _C_perTileNnz.back() << "\n\n";

        thrustHvecPin<ValueType> Cvals = Ctiles_vals;
        thrustHvecPin<uint8_t> CrowColIdx = Ctiles_rowColIdx;

        outfile << "VALS\n";
        for(auto v: Cvals) outfile << v << " ";
        outfile << "\nROWCOLIDX\n";
        for(auto rc: CrowColIdx) {
            // std::bitset<8> b (rc);
            // outfile << b.to_string().c_str() << " ";

            auto r = rc >> 4;
            auto c = rc & 0xF;
            outfile << r << "-" << c << " ";
        }

        outfile.close();
    });
#endif

#ifdef DEBUG_7
    std::jthread DEBUG_7([&](){
        char const *filename = "../src/DEBUG/DEBUG_C_7";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        auto print = [&outfile] (auto v) { outfile << v << " "; };
        thrustHvec<TileCSR_C<ValueType>> hCtiles(Ctiles.size());
        thrustHvec<int> hCperTileNnz(_C_perTileNnz.size());
        thrust::copy(ASYNC_EXEC_POLICY(STREAM_C), Ctiles.begin(), Ctiles.end(), hCtiles.begin());
        thrust::copy(ASYNC_EXEC_POLICY(STREAM_C), _C_perTileNnz.begin(), _C_perTileNnz.end(), hCperTileNnz.begin());
        cudaStreamSynchronize(STREAM_C);
        
        outfile << "]\n" << std::endl;
        outfile << "Ctiles: [(y, x)" << std::endl;
        for(int i = 0; i < hCtiles.size(); ++i) {
            printInfo2(outfile, hCtiles[i], 256);
        }
        outfile << "]" << std::endl;

        outfile << "\n\nperTileNnz:\n";
        std::for_each(hCperTileNnz.begin(), hCperTileNnz.end(), print);

        outfile.close();
    });
#endif

    cudaStreamSynchronize(STREAM_C);
    std::cout << "C nnz: " << _C_perTileNnz.back() << "\n";

    std::quick_exit(0);
    // std::atexit([]{cudaDeviceReset();});
    return 0; // <--------------------------------------------------------------------------------------------------------------------------------

    // rmm::device_vector<int> Crows(_C_perTileNnz.back(), SPGEMM_STREAM_ALLOCATOR_INT(STREAM_C));
    // rmm::device_vector<int> Ccols(_C_perTileNnz.back(), SPGEMM_STREAM_ALLOCATOR_INT(STREAM_C));
    // rmm::device_vector<ValueType> Cvals(_C_perTileNnz.back(), SPGEMM_STREAM_ALLOCATOR_VALUETYPE(STREAM_C));

    // dim3 threads_sC {tileSize * tileSize};
    // dim3 blocks_sC {Ctiles.size()};

    // sanitize_C<<<blocks_sC, threads_sC, 0, STREAM_C>>>
    // (
    //     Crows.data().get(), 
    //     Ccols.data().get(), 
    //     Cvals.data().get(), 
    //     Ctiles.data().get(), 
    //     _C_tilePtr.data().get(),
    //     _C_tilePtr.size(), 
    //     _C_tileColIdx.data().get(), 
    //     _C_perTileNnz.data().get()
    // );

#ifdef DEBUG_8
    std::jthread DEBUG_8([&](){
        char const *filename = "../src/DEBUG/DEBUG_C_8";
        std::ofstream outfile;
        outfile.open(filename, std::ios::out);

        auto print = [&outfile] (auto v) { outfile << v << " "; };
        
        thrustHvec<int> hCrows(Crows.size());
        thrustHvec<int> hCcols(Ccols.size());
        thrustHvec<ValueType> hCvals(Cvals.size());

        cudaStreamSynchronize(STREAM_C);
        thrust::copy(Crows.begin(), Crows.end(), hCrows.begin());
        thrust::copy(Ccols.begin(), Ccols.end(), hCcols.begin());
        thrust::copy(Cvals.begin(), Cvals.end(), hCvals.begin());

        outfile << "MATRIX C -- RESULT\n";
        outfile << "rows:\n";
        std::for_each(hCrows.begin(), hCrows.end(), print);
        outfile << "\n\ncols:\n";
        std::for_each(hCcols.begin(), hCcols.end(), print);
        outfile << "\n\nvals:\n";
        std::for_each(hCvals.begin(), hCvals.end(), print);

        outfile.close();
    });
#endif

    // streams are destroyed by rmm
}
