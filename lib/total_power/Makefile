prefix ?= /usr/local
libdir ?= $(prefix)/lib
includedir ?= $(prefix)/include
bindir ?= $(prefix)/bin

all: total_power.so test_total_power.o
	gcc -L /usr/local/cuda-7.0/lib64 -lcudart -lstdc++ -o flag_power total_power.so test_total_power.o

total_power.so: total_power.cu total_power.h
	nvcc -c -o total_power.so --shared total_power.cu

test_total_power.o: test_total_power.c total_power.h
	gcc -I /usr/local/cuda-7.0/include -c -o test_total_power.o test_total_power.c 

clean:
	rm -f *.o
	rm -f flag_power

install: total_power.h total_power.so
	mkdir -p $(includedir)
	cp total_power.h $(includedir)
	mkdir -p $(libdir)
	cp total_power.so $(libdir)
	mkdir -p $(bindir)
	cp flag_power $(bindir)
