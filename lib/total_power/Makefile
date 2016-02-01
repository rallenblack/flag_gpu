all: total_power.o test_total_power.o
	gcc -L /usr/local/cuda-7.0/lib64 -lcudart -lstdc++ -o run total_power.o test_total_power.o

total_power.o: total_power.cu total_power.h
	nvcc -c -o total_power.o total_power.cu

test_total_power.o: test_total_power.c total_power.h
	gcc -I /usr/local/cuda-7.0/include -c -o test_total_power.o test_total_power.c 

clean:
	rm -f *.o
	rm -f run
