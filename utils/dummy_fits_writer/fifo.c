//# Copyright (C) 2015 Associated Universities, Inc. Washington DC, USA.
//#
//# This program is free software; you can redistribute it and/or modify
//# it under the terms of the GNU General Public License as published by
//# the Free Software Foundation; either version 2 of the License, or
//# (at your option) any later version.
//#
//# This program is distributed in the hope that it will be useful, but
//# WITHOUT ANY WARRANTY; without even the implied warranty of
//# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
//# General Public License for more details.
//#
//# You should have received a copy of the GNU General Public License
//# along with this program; if not, write to the Free Software
//# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
//#
//# Correspondence concerning GBT software should be addressed as follows:
//# GBT Operations
//# National Radio Astronomy Observatory
//# P. O. Box 2
//# Green Bank, WV 24944-0002 USA

#include <poll.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fifo.h"

const int MAX_CMD_LEN = 64;

// int fifo_fd = -1;

int open_fifo(char *fifo_loc)
{
	int fifo_fd = open(fifo_loc, O_RDONLY | O_NONBLOCK);
	if (fifo_fd<0)
	{
		fprintf(stderr, "vegas_fits_writer: Error opening control fifo %s\n", fifo_loc);
		perror("open");
// 		exit(1);
	}

    fprintf(stderr, "FIFO created with fd: %d\n", fifo_fd);
	return fifo_fd;
}

cmd_t check_cmd(int fifo_fd)
{
    // fprintf(stderr, "Checking FIFO with fd: %d\n", fifo_fd);
	char cmd[MAX_CMD_LEN];

	// fprintf(stderr, "fifo_fd: %d\n", fifo_fd);

	struct pollfd pfd[2];
        pfd[1].fd = fifo_fd;
        pfd[1].events = POLLIN;
        pfd[0].fd = fileno(stdin);
        pfd[0].events = POLLIN;
		// ?, num file desc, timeout
        int rv = poll(pfd, 2, 1000);
        if (rv==0)
        {
// 			fprintf(stderr, "rv == 0 :(\n");
            return INVALID; //????
        }
        else if (rv<0)
        {
            if (errno!=EINTR)
            {
                perror("poll");
            }
//             fprintf(stderr, "rv < 0 :(\n");
            return INVALID; //????
        }

        // clear the command
        memset(cmd, 0, MAX_CMD_LEN);
		int i;
        for (i=0; i<2; ++i)
        {
            rv = 0;
            if (pfd[i].revents & POLLIN)
            {
                if (read(pfd[i].fd, cmd, MAX_CMD_LEN-1)<1)
				{
// 					fprintf(stderr, "read failed :(\n");
// 					perror("read");
//                     return INVALID;
				}
                else
                {
// 					fprintf(stderr, "read success :(\n");
                    rv = 1;
                    break;
                }
            }
        }

        //if (pfd[0].revents==POLLHUP)
	//		fprintf(stderr, "POLLHUP :(\n");

		if (rv==0)
        {
// 			fprintf(stderr, "rv == 0 again :(\n");
            return INVALID;
        }
        else if (rv<0)
        {
// 			fprintf(stderr, "rv < 0 again :(\n");
            if (errno==EAGAIN)
            {
                return INVALID;
            }
            else
            {
                perror("read");
                return INVALID;
            }
        }

        // Truncate at newline
        // TODO: allow multiple commands in one read?
        char *ptr = strchr(cmd, '\n');
        if (ptr!=NULL)
        {
            *ptr='\0';
        }

        // Process the command
        if (strncasecmp(cmd,"START",MAX_CMD_LEN)==0)
        {
			return START;
        }
        else if (strncasecmp(cmd,"STOP",MAX_CMD_LEN)==0)
		{
			return STOP;
		}
		else if (strncasecmp(cmd,"QUIT",MAX_CMD_LEN)==0)
		{
			return QUIT;
		}
        else
        {
            // Unknown command
            return INVALID;
        }


}
