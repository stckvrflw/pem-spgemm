# pem-spgemm
#### **BETA**
Final Assignment Project - SpGEMM algorithm in CUDA  
By Petrus E. Manurung  
2025

An Improved Sparse General Matrix-Matrix Multiplication (SpGEMM) algorithm.  
Improving upon TileSpGEMM by eliminating atomics and better cache utilization on step 2 and step 3.  
Another improvement includes native GPU implementation of conversion from .mtx to Tiled CSR intermediate format. 

Libraries used:
* [thrust][thrust]
* [rmm][rapidsrmm]
* [fast_matrix_market][fmm]
* [nsparse][nsparse]

Other resources:
* Sparse matrices from [suiteSparse][suitesparse]

Reference:
1. [TileSpGEMM -- **Niu et al.**](https://doi.org/10.1145/3503221.3508431)


Environment:
* CPU       : 11th Gen Intel(R) Core(TM) i7-11800H
* GPU       : NVIDIA Corporation GA104M [GeForce RTX 3080 Mobile / Max-Q 8GB/16GB]
* OS        : Gentoo Linux
* Kernel    : 6.13.8-zen1
* CUDA      : 12.8
* driver    : 570.144
* gcc       : 14.2.1 20241221

How to compile:  
1. clone this repository
2. get rapidsrmm v24.12.00 from [rapidsrmm] and extract to pem-spgemm (cloned repo)
3. get fastmatrixmarket v1.7.6 from [fmm] and extract to pem-spgemm (cloned repo)
4. run "make"

How to use:
* A^2   : ./pemspgemm "path-to-.mtx-file" [0/1] 
* A*At  : ./pemspgemm "path-to-.mtx-file" [0/1] 1  
*** 0 to skip saving result (in COO) to file, 1 to save to /tmp  
*** since /tmp is in RAM, make sure there is enough space.  
(e.g. result from A^2 of webbase-1M can cost more than 1.5GiB)  
*** no quote on path to mtx-file  

To reproduce: GPU with sm_86  
if using different GPU, change the "code" part in NVCC_FLAGS in the Makefile.  
Keep "compute_61" unchanged.

Benchmark result is saved in 'pemspgemm_benchmark_result.csv' file  
header for the csv:  
matrix,flop,C_nnz,compression_ratio,A_conversion_kernel_time,B_conversion_kernel_time,total_conversion_overhead_time,step1_time,step2_time,step3_time,pem_spgemm_time,pem_spgemm_kernel_time,pem_spgemm_malloc_time,Gflops

[ansorge]: https://github.com/RichardAns/CUDA-Programs
[thrust]: https://developer.nvidia.com/thrust
[rapidsrmm]: https://github.com/rapidsai/rmm
[cusparse]: https://developer.nvidia.com/cusparse
[fmm]: https://github.com/alugowski/fast_matrix_market
[suitesparse]: https://sparse.tamu.edu
[nsparse]: https://github.com/EBD-CREST/nsparse