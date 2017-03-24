-include Makefile.local

prefix ?= /usr/local
libdir ?= $(prefix)/lib
includedir ?= $(prefix)/include
bindir ?= $(prefix)/bin
CUDA_DIR ?= $(prefix)/cuda

#CUDA_DIR ?= /usr/local/cuda
#prefix ?= tmp
#libdir ?= $(prefix)/lib
#includedir ?= $(prefix)/include
#bindir ?= $(prefix)/bin
#CUDA_DIR ?= /Developer/NVIDIA/CUDA-6.5

INCLUDES = -I$(CUDA_DIR)/include
CUDA_LIBDIR = -L$(CUDA_DIR)/lib
#INCLUDES = -I/Developer/NVIDIA/CUDA-6.5/include
#CUDA_LIBDIR = -L/Developer/NVIDIA/CUDA-6.5/lib

CLIBDIR = -L/usr/local/lib
CU_LIB  = -lcufft -lcudart
C_LIB = -ldl -lutil

LINKER = -Xlinker -export-dynamic

ifneq ($(strip $(OSTYPE)),osx)
CUDA_LIBDIR ?= $(CUDA_DIR)/lib64
else
CUDA_LIBDIR ?= $(CUDA_DIR)/lib
endif

NVCC = $(CUDA_DIR)/bin/nvcc

all: libflagpfb.so

libflagpfb.so: pfb.o kernels.o
	$(NVCC) $(INCLUDS) -Xcompiler -fPIC -Xcompiler -D_REENTRANT -O3 pfb.o kernels.o -o libflagpfb.so $(CUDA_LIBDIR) $(CU_LIB) $(CLIBDIR) $(C_LIB) $(LINKER) --shared

pfb.o: kernels.cu kernels.h
	$(NVCC) $(INCLUDES) $(CUDA_LIBDIR) $(CU_LIB) -c -o pfb.o -Xcompiler -fPIC -Xcompiler -D_REENTRANT -O3 pfb.cu

kernels.o: kernels.cu kernels.h
	$(NVCC) $(INCLUDES) $(CUDA_LIBDIR) $(CU_LIB) -c -o kernels.o kernels.cu -Xcompiler -fPIC -Xcompiler -D_REENTRANT -O3

clean:
	rm -f *.o
	rm -f *.so

install: kernels.o pfb.o libflagpfb.so
	mkdir -p $(includedir)
	cp pfb.h $(includedir)
	cp kernels.h $(includedir)
	mkdir -p $(libdir)
	cp libflagpfb.so $(libdir)
	#mkdir -p $(bindir)
	#cp cublas_main $(bindir)
