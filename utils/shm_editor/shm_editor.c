#include <sys/shm.h>
#include <sys/ipc.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include "fitshead.h"
#define INSTANCE_ID 0
#define TOTAL_SIZE (2880*64)
#define KEY 0x42000001



int main() {

    int proj_id = ((INSTANCE_ID & 0x3f)|0x40);
    printf("Project ID = 0x%08x\n", proj_id);

    key_t key = -1;
    char * status_key = getenv("HASHPIPE_STATUS_KEY");
    if (status_key) {
        key = strtoul(status_key, NULL, 0);
    } else {
        char * keyfile = getenv("HASHPIPE_KEYFILE");
        if (!keyfile) {
            keyfile = getenv("HOME");
            if (!keyfile) {
                keyfile = "/tmp";
            }
        }
        key = ftok(keyfile, proj_id);
    }

    int shmid = shmget(KEY, TOTAL_SIZE, 0666);
    if (shmid == -1) {
	printf("shmget error! ");
	if (errno == EACCES) printf("Permission Denied!\n");
        if (errno == EINVAL) printf("Invalid size specified\n");
        if (errno == ENFILE) printf("System limit of open files reached\n");
        if (errno == ENOENT) printf("No segement exists for given key\n");
        return -1;
    }
    printf("Obtained Shared Memory ID %d...\n", shmid);

    char * buffer;
    buffer = shmat(shmid, NULL, 0);

    char integ_status[17];

    int keep_looping = 1;
    while (keep_looping) {
        hgets(buffer, "INTSTAT", 16, integ_status);
        printf("INTSTAT = %s\n", integ_status);
        if (strcmp(integ_status, "stop") == 0) {
            int input = -1;
            printf("New Scan? [yes = 1, no = 0, exit = -1]: ");
            scanf("%d", &input);
            switch(input) {
    	        case -1:
                    keep_looping = 0;
		    break;
		case 0:
		    break;
	        case 1:
		    hputs(buffer, "INTSTAT", "start");
		    break;
		default:
		    break;
	    }
        } else {
            int input = -1;
            printf("Stop Scan? [yes = 1, no = 0, exit = -1]: ");
            scanf("%d", &input);
            switch(input) {
    	        case -1:
                    keep_looping = 0;
		    break;
		case 0:
		    break;
	        case 1:
		    hputs(buffer, "INTSTAT", "stop");
		    break;
		default:
		    break;
	    }
	}
    }
}
