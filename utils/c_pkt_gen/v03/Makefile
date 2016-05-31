CC = gcc
FLAGS = -lpthread -lm

all: bfpacketizer0.3

udp_recv:
	$(CC) $(FLAGS) -o udp_recv udp_recv.c

udp_send:
	$(CC) $(FLAGS) -o udp_send udp_send.c

udp_sendtst:
	$(CC) $(FLAGS) -o udp_sendtst udp_sendtst.c

bfpacketizer0.1:
	$(CC) $(FLAGS) -o bfpacketizer0.1 bfpacketizer0.1.c

bfpacketizer0.2:
	$(CC) $(FLAGS) -o bfpacketizer0.2 bfpacketizer0.2.c

bfpacketizer0.3: bfpacketizer0.3.c bfpacketizer_params.h
	$(CC) $(FLAGS) -o bfpacketizer0.3 bfpacketizer0.3.c

bfpacketrecv0.1:
	$(CC) $(FLAGS) -o bfpacketrecv0.1 bfpacketrecv0.1.c

bfpacketrecv0.2:
	$(CC) $(FLAGS) -o bfpacketrecv0.2 bfpacketrecv0.2.c

