# pem-spgemm
#### **WORK IN PROGRESS**
Final Assignment Project - SpGEMM algorithm in CUDA

Original idea from [1]

A Sparse General Matrix-Matrix Multiplication (SpGEMM) implementation.  
Implemented it my own way from scratch with custom optimization and changes in CUDA C++20.
Makes extensive use of cooperative groups.

Libraries used:
* [cx.h][ansorge]
* [thrust][thrust]
* [rmm][rapidsrmm]
* [cusparse][cusparse]
* [fast_matrix_market][fmm]

Other resources:
* Sparse matrices from [suiteSparse][suitesparse]

Papers:
1. [TileSpGEMM -- **Niu et al.**](https://doi.org/10.1145/3503221.3508431)


Environment:
* CPU       : 11th Gen Intel(R) Core(TM) i7-11800H
* GPU       : NVIDIA Corporation GA104M [GeForce RTX 3080 Mobile / Max-Q 8GB/16GB]
* OS        : Gentoo Linux
* Kernel    : 6.12.16-gentoo
* CUDA      : 12.8
* driver    : 570.124.04
* gcc       : 14.2.1 20241221

[ansorge]: https://github.com/RichardAns/CUDA-Programs
[thrust]: https://developer.nvidia.com/thrust
[rapidsrmm]: https://github.com/rapidsai/rmm
[cusparse]: https://developer.nvidia.com/cusparse
[fmm]: https://github.com/alugowski/fast_matrix_market
[suitesparse]: https://sparse.tamu.edu


#### TODO:
* finish the project