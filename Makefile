# ENVIRONMENT_PARAMETERS - edit as needed
CUDA_INSTALL_PATH=/opt/cuda
FMM_PATH:=${CURDIR}/fast_matrix_market-1.7.6# fast_matrix_market path
RMM_PATH:=${CURDIR}/rmm-24.12.00# rapids rmm path
CC=$(CUDA_INSTALL_PATH)/bin/nvcc
HOST_CC=/usr/x86_64-pc-linux-gnu/gcc-bin/14/g++

# CUDA_PARAMETERS
NVCC_FLAGS = -forward-unknown-to-host-compiler \
 --expt-relaxed-constexpr --expt-extended-lambda \
 -std=c++20 \
 -Xcompiler -O3 -O3 \
 -w -arch=compute_61 -code=sm_86 -gencode=arch=compute_61,code=sm_86 \
 -ccbin $(HOST_CC)

# includes
INCLUDES = -I$(CUDA_INSTALL_PATH)/ncu/host/target-linux-x64/nvtx/include \
 -I$(CUDA_INSTALL_PATH)/include \
 -I$(RMM_PATH)/include \
 -I$(FMM_PATH)/include \
 -I./Common \
 -I./NSPARSE
 
# Defines
DEFINES =  -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \
 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA \
 -DTHRUST_DISABLE_ABI_NAMESPACE \
 -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP \
 -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR

CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart
LIBS = $(CUDA_LIBS)

OPTS = -DREPEAT=10

make:
	$(CC) $(NVCC_FLAGS) $(DEFINES) -Xcompiler -mfma spgemm.cu -o pemspgemm $(INCLUDES) $(LIBS) $(OPTS) 