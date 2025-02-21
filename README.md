# pem-spgemm
#### **WORK IN PROGRESS**
Final Assignment Project - SpGEMM algorithm in CUDA

Original idea from [1]

A Sparse General Matrix-Matrix Multiplication (SpGEMM) implementation.  
Implemented it my own way from scratch with custom optimization and changes in CUDA C++20.

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
* OS        : Gentoo Linux
* Kernel    : 6.12.10-zen1
* CUDA      : 12.6
* gcc       : 13.3.1_p20241220 p2

[ansorge]: https://github.com/RichardAns/CUDA-Programs
[thrust]: https://developer.nvidia.com/thrust
[rapidsrmm]: https://github.com/rapidsai/rmm
[cusparse]: https://developer.nvidia.com/cusparse
[fmm]: https://github.com/alugowski/fast_matrix_market
[suitesparse]: https://sparse.tamu.edu


#### TODO:
* finish the project