#include <stdio.h>
#include <stdlib.h>

#include <string.h>		/* for strncopy(), memcpy(), strerror()*/
#include <sys/types.h>  /* for open()  */
#include <sys/stat.h>	/* for open()  */
#include <sys/time.h>
#include <fcntl.h>		/* for open()  */
#include <unistd.h>		/* for close() */
#include <errno.h>		/* for errno   */  

#include <getopt.h>		/* for option parsing */

#include "pfb.h"
#include "helper.h"
#include "tools/tools.h"