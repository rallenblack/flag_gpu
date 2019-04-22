-include Makefile.local

prefix ?= /usr/local
libdir ?= $(prefix)/lib
includedir ?= $(prefix)/include
bindir ?= $(prefix)/bin
CUDA_DIR ?= $(prefix)/cuda
ifneq ($(strip $(OSTYPE)),osx)
CUDA_LIBDIR ?= $(CUDA_DIR)/lib64
else
CUDA_LIBDIR ?= $(CUDA_DIR)/lib
endif

NVCC = $(CUDA_DIR)/bin/nvcc

all: libflagbeamformer.so cublas_main.o
	gcc -L $(CUDA_LIBDIR) -lstdc++ -lcudart -lcublas -o cublas_main libflagbeamformer.so cublas_main.o

cublas_beamformer.o: cublas_beamformer.cu cublas_beamformer.h
	$(NVCC) -c -o cublas_beamformer.o -Xcompiler -fPIC -Xcompiler -D_REENTRANT -O3 cublas_beamformer.cu

libflagbeamformer.so: cublas_beamformer.o
	$(NVCC) -Xcompiler -fPIC -Xcompiler -D_REENTRANT -O3 cublas_beamformer.o -o libflagbeamformer.so -L$(CUDA_LIBDIR) --shared

cublas_main.o: cublas_main.cu cublas_beamformer.h
	$(NVCC) -I $(CUDA_DIR)/include -c -o cublas_main.o cublas_main.cu

clean:
	rm -f *.o
	rm -f *.so
	rm -f cublas_main

install: cublas_main cublas_beamformer.h libflagbeamformer.so
	mkdir -p $(includedir)
	cp cublas_beamformer.h $(includedir)
	mkdir -p $(libdir)
	cp libflagbeamformer.so $(libdir)
	mkdir -p $(bindir)
	cp cublas_main $(bindir)
