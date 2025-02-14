# pem-spgemm
#### **WORK IN PROGRESS**
Final Assignment Project - SpGEMM algorithm in CUDA

Original idea from [TileSpGEMM -- **Niu et al.**](https://doi.org/10.1145/3503221.3508431)

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


[ansorge]: https://github.com/RichardAns/CUDA-Programs
[thrust]: https://developer.nvidia.com/thrust
[rapidsrmm]: https://github.com/rapidsai/rmm
[cusparse]: https://developer.nvidia.com/cusparse
[fmm]: https://github.com/alugowski/fast_matrix_market
[suitesparse]: https://sparse.tamu.edu


#### TODO:
* include CMake config
* finish the project