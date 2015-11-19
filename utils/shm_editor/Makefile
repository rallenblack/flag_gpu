all: hget.o hput.o shm_editor.o
	gcc -lm shm_editor.o hget.o hput.o

hget.o: hget.c fitshead.h
	gcc -c hget.c -o hget.o

hput.o: hput.c fitshead.h
	gcc -c hput.c -o hput.o

shm_editor.o: shm_editor.c
	gcc -c shm_editor.c -o shm_editor.o
