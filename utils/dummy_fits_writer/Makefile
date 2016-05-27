all: fifo.o dummy_fits_writer.o
	gcc -o dummy_fits_writer fifo.o dummy_fits_writer.o

fifo.o: fifo.c fifo.h
	gcc -o fifo.o -c fifo.c

dummy_fits_writer.o: dummy_fits_writer.c fifo.h
	gcc -o dummy_fits_writer.o -c dummy_fits_writer.c

install: dummy_fits_writer
	cp dummy_fits_writer /usr/local/bin

clean:
	rm -f *.o
	rm -f dummy_fits_writer
