#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include "fifo.h"
#include <unistd.h>

static int quit = 0;

// Control-C handler
static void cc(int sig) {
    quit = 1;
}

int main(int argc, char * argv[]) {
    signal(SIGINT, cc); // Connect Ctrl+C to handler
    int cmd = INVALID;
    int fits_fifo_id = open_fifo("/tmp/fits_bogus.fifo");
    while(quit != 1) {
        cmd = check_cmd(fits_fifo_id);
        switch(cmd) {
            case START:
                printf("dummy_fits_writer: START\n");
                break;
            case STOP:
                printf("dummy_fits_writer: STOP\n");
                break;
            case QUIT:
                printf("dummy_fits_writer: QUIT\n");
                quit = 1;
                break;
        }
        sleep(0.2);
    }
    return 1;
}
