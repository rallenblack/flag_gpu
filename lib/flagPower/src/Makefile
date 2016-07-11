prefix ?= /usr/local
libdir ?= $(prefix)/lib
includedir ?= $(prefix)/include
bindir ?= $(prefix)/bin

all: libflagpower.so test_total_power.o
	gcc -L /usr/local/cuda/lib64 -lstdc++ -lcudart -o flagpower libflagpower.so test_total_power.o

total_power.o: total_power.cu total_power.h
	nvcc -c -o total_power.o -Xcompiler -fPIC -Xcompiler -D_REENTRANT -O3 total_power.cu

libflagpower.so: total_power.o
	nvcc -Xcompiler -fPIC -Xcompiler -D_REENTRANT -O3 total_power.o -o libflagpower.so -L/usr/local/cuda/lib64 --shared

test_total_power.o: test_total_power.c total_power.h
	gcc -I /usr/local/cuda/include -c -o test_total_power.o test_total_power.c 

clean:
	rm -f *.o
	rm -f *.so
	rm -f flagpower

install: total_power.h libflagpower.so
	mkdir -p $(includedir)
	cp total_power.h $(includedir)
	mkdir -p $(libdir)
	cp libflagpower.so $(libdir)
	mkdir -p $(bindir)
	cp flagpower $(bindir)
