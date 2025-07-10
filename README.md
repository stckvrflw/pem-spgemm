# pem-spgemm
#### **BETA**
Final Assignment Project - SpGEMM algorithm in CUDA

Inspirations from [1]

An Improved Sparse General Matrix-Matrix Multiplication (SpGEMM) algorithm.  
Improving upon TileSpGEMM by eliminating atomics and better cache utilization on step 2 and step 3.

Libraries used:
* [cx.h][ansorge]
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
TODO wrap up the build system;  
The program works. But dependencies might need to be prepared manually.  
To reproduce: GPU with sm_86 (edit CMakeFiles.txt)  

How to use:
* A^2   : ./spgemm "path-to-.mtx-file" [0/1] 
* A*At  : ./spgemm "path-to-.mtx-file" [0/1] 1  
*** 0 to skip saving result (in COO) to file, 1 to save to file in /tmp
*** no quote on path to mtx-file

[ansorge]: https://github.com/RichardAns/CUDA-Programs
[thrust]: https://developer.nvidia.com/thrust
[rapidsrmm]: https://github.com/rapidsai/rmm
[cusparse]: https://developer.nvidia.com/cusparse
[fmm]: https://github.com/alugowski/fast_matrix_market
[suitesparse]: https://sparse.tamu.edu
[nsparse]: https://github.com/EBD-CREST/nsparse