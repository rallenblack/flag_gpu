#ifndef __HELPER_H
#define __HELPER_H

#include <stdio.h>
#include <stdlib.h>

// Had to copy these over form pfb.h since they are part of a load memory helper function most likely not needed in full pfb implementation
#include <string.h>		/* for strncopy(), memcpy(), strerror()*/
#include <sys/types.h>  /* for open()  */
#include <sys/stat.h>	/* for open()  */
#include <fcntl.h>		/* for open()  */
#include <unistd.h>		/* for close() */
#include <errno.h>		/* for errno   */

void printUsage(const char* progName);
int loadData(char* f, char* inputData, int size);


#endif