prefix ?= /usr/local
libdir ?= $(prefix)/lib
includedir ?= $(prefix)/include
bindir ?= $(prefix)/bin

all: libflagbeamformer.so test_beamformer.o
	gcc -L /usr/local/cuda-7.0/lib64 -lstdc++ -lcudart -o test_beamformer libflagbeamformer.so test_beamformer.o

flag_beamformer.o: flag_beamformer.cu flag_beamformer.h
	nvcc -c -o flag_beamformer.o -Xcompiler -fPIC -Xcompiler -D_REENTRANT -O3 flag_beamformer.cu

libflagbeamformer.so: flag_beamformer.o
	nvcc -Xcompiler -fPIC -Xcompiler -D_REENTRANT -O3 flag_beamformer.o -o libflagbeamformer.so -L/usr/local/cuda-7.0/lib64 --shared

test_beamformer.o: test_beamformer.cu flag_beamformer.h
	nvcc -I /usr/local/cuda-7.0/include -c -o test_beamformer.o test_beamformer.cu

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
