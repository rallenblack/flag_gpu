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

all: libflagbeamformer.so test_beamformer.o
	gcc -L $(CUDA_LIBDIR) -lstdc++ -lcudart -o test_beamformer libflagbeamformer.so test_beamformer.o

flag_beamformer.o: flag_beamformer.cu flag_beamformer.h
	$(NVCC) -c -o flag_beamformer.o -Xcompiler -fPIC -Xcompiler -D_REENTRANT -O3 flag_beamformer.cu

libflagbeamformer.so: flag_beamformer.o
	$(NVCC) -Xcompiler -fPIC -Xcompiler -D_REENTRANT -O3 flag_beamformer.o -o libflagbeamformer.so -L$(CUDA_LIBDIR) --shared

test_beamformer.o: test_beamformer.cu flag_beamformer.h
	$(NVCC) -I $(CUDA_DIR)/include -c -o test_beamformer.o test_beamformer.cu

clean:
	rm -f *.o
	rm -f *.so
	rm -f test_beamformer

install: test_beamformer flag_beamformer.h libflagbeamformer.so
	mkdir -p $(includedir)
	cp flag_beamformer.h $(includedir)
	mkdir -p $(libdir)
	cp libflagbeamformer.so $(libdir)
	mkdir -p $(bindir)
	cp test_beamformer $(bindir)
