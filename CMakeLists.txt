cmake_minimum_required(VERSION 3.30.5)
list(APPEND CMAKE_PREFIX_PATH "/opt/cuda/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_COMPILER "/usr/x86_64-pc-linux-gnu/gcc-bin/13/gcc")
project(ta2 LANGUAGES CUDA CXX)


find_package(CUDAToolkit REQUIRED)

# set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS_DEBUG "-Xptxas -O0 -g3 -G -Xcompiler -mfma -gencode=arch=compute_61,code=sm_86 --device-debug --expt-relaxed-constexpr --expt-extended-lambda -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g3")
set(CMAKE_CUDA_FLAGS_RELEASE "-Xptxas -O3,-v -Xcompiler -mfma -DNDEBUG --expt-relaxed-constexpr --expt-extended-lambda -gencode=arch=compute_61,code=sm_86 -march=native -O3 -DREPEAT=5")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -std=c++20")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-Xptxas -O2,-v -Xcompiler -mfma -march=native -O2 -g -gencode=arch=compute_61,code=sm_86 -DNDEBUG --expt-relaxed-constexpr --expt-extended-lambda -lineinfo")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-O1 -DNDEBUG --expt-extended-lambda")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

set(CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_SEPARABLE_COMPILATION ON)
add_compile_definitions(THRUST_IGNORE_CUB_VERSION_CHECK)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ansorge)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ansorge)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mmio)
add_library(NSPARSE SHARED ${CMAKE_CURRENT_SOURCE_DIR}/NSPARSE/spgemm_nsparse_kernel.cu)
target_include_directories(NSPARSE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/NSPARSE)


include("./CPM.cmake")

CPMAddPackage(NAME rmm
              GITHUB_REPOSITORY rapidsai/rmm
              GIT_TAG v24.12.01
              SYSTEM Off)


include(FetchContent)
FetchContent_Declare(
        fast_matrix_market
        GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
        GIT_TAG main
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)

add_executable(spgemm spgemm.cu)

target_link_libraries(spgemm fast_matrix_market::fast_matrix_market ${CUDA_cusparse_LIBRARY} rmm::rmm)
set_target_properties(spgemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)