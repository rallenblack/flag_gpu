/* udp_send.c developed by Paul Demorest 12-15-2007
 * was modified to creat bfpacketizerx.x.c. Parameters
 * used by this program are defined in bfpacketizer_params.h
 *
 * V0.1: emulates 1 Roach board sending packets as defined
 * by BYU to Kmax GPU machines. The ipaddress of the GPUs
 * are defined in bfpacketizer_params.h
 *
 * Anish & Arnab July 2015 
 *
 * V0.2: emulates 5 Roach board sending packets as defined
 * by BYU to Kmax GPU machines. The packets are send by 5 threads. 
 * The ipaddress of the GPUs are defined in bfpacketizer_params.h
 *
 * Arnab & Anish Aug 2015
 *
 * V0.3: Changes made --
 * 1. Default value of total_data changed to 1.0 GB
 * 2. mcnt changed to m to make the code more readable. 
 * 3. packet_count changed to mcnt to reflect that it
 * is not the count of packets that are sent from a roach.
 *
 * Anish Aug 19, 2015
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<complex.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <signal.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <pthread.h>

#include "bfpacketizer_params.h"
#define TOTAL_DATA 1.0

unsigned long long MCNT = 0x0; //Global MCNT for thread synch 

struct hostent *hh[Nxengines];
struct sockaddr_in ip_addr[Nxengines];
int slen;
int sock[Nfengines];
float wait_cyc=0;
float total_data = TOTAL_DATA;
int StartSendingPacket=0;
int packet_size = PACKET_SIZE;
int endian=0;

int run=1;
void cc(int sig) { run=0; }

void usage() {
    fprintf(stderr,
            "Usage: bfpacketizer0.3 (options) \n"
            "Options:\n"
            "  -p nn, --port=nn         Port number (%d)\n"
            "  -s nn, --packet-size=nn  Packet size, bytes (%d)\n"
            "  -d nn, --total-data=nn   Total amount to send, GB (%.1f)\n"
            "  -q, --quiet              More compact output\n"
            "  -w nn, --wait=nn         Wait 1000*nn cycles (0)\n"
            "  -c, --sine_channel       Channel number for sine wave (0 to 499)\n"
            "  -a, --sine_phase         Phase of the sine wave in rad \n"
            "  -e, --endian             Byte-swap seq num\n"
            , PORT_NUM, PACKET_SIZE, TOTAL_DATA);
}


//Thread Variable for the packetizer
struct send_packet_data{
	unsigned char m; 
	unsigned char cal; 
	int Kmax; 
        int SineCh;
	float SinePh;
	int SinePer; 
        int UseGlobalMCNT;

};

//Byte swap
void byte_swap(unsigned long long *d) {
    	unsigned long long tmp;
   	char *ptr1, *ptr2;
    	ptr1 = (char *)d;
    	ptr2 = (char *)&tmp + 7;
    	int i;
    	for (i=0; i<8; i++) {
        	ptr1 = (char *)d + i;
        	ptr2 = (char *)&tmp + 7 - i;
        	memcpy(ptr2, ptr1, 1);
   	}
    	*d = tmp;
}

//Header generation function
void header_gen(unsigned char *packet_header, unsigned long long MCNT, 
                unsigned char m, unsigned char k, unsigned char cal)
{
	int p;
        unsigned long long header;
	cal = (cal & 0xF);
	//Fill header with MCNT, cal, m and k
	header = MCNT;
	header = ((header << 4) | cal);
	header = ((header << 8) | m);
	header = ((header << 8) | k);
	
	//Fill packet_header with the header. 
	for (p = 7;p >= 0; p--){
		packet_header[p] = (unsigned char)(header);
		header = (header >> 8);
	}		
}

char quantize(double trig)
{
	return (char) round(trig*127);
}

//Payload generation 
void payload_gen(char *data, unsigned char K, 
                 int SineCh, float SinePh, int SinePer)
{
	int t, i, cc, ch;
	int Nin_per_f = Nin_per_f(); /* Number of inputs per ROACH */
        int nbins = Nbins;
        int ntime_per_packet = Ntime_per_packet;
        int ContChan = 5; /* Contiguous channels to be packed */
        int ChanInc=100; /* Contiguous channels are incremented in this step */ 
        int SetOfChan, DatIndx, DatSizeForAtime, PaylSize, 
            DatSizeForNin,chan,DatSizeForContChan;

        SetOfChan = nbins/ChanInc; /* Number of contiguous set of channels in payload */
        DatSizeForAtime = Nin_per_f*ContChan*SetOfChan*2; /*Total data size in byte 
                                                            for a time stamp */
        DatSizeForContChan = Nin_per_f*ContChan*2;
        DatSizeForNin = Nin_per_f * 2; /*Total data size in byte for all inputs per ROACH */
        PaylSize = DatSizeForAtime * ntime_per_packet; /*Total payload size in bytes */

	for(t = 0; t < ntime_per_packet; t++){ /* Loop through time samples */
	   for(cc = 0; cc < SetOfChan; cc++){  /* Loop through contiguous set of channels*/
	      for(ch = 0; ch < ContChan; ch++){ /* Loop through contiguous channels */
                 chan = ContChan*K + ch + cc*ChanInc; /* Channel number */ 
		 for(i = 0; i < Nin_per_f; i++){
                     DatIndx = i*2 + ch*DatSizeForNin + cc*DatSizeForContChan + t*DatSizeForAtime;
		     //printf("%d %d\n",chan,DatIndx);			
                     if (chan == SineCh){  
			data[DatIndx] = quantize(cos(SinePh));		
			data[DatIndx+1] = quantize(sin(SinePh));		
		     }
		     else{
			data[DatIndx] = 0;		
			data[DatIndx+1] = 0;		
		     }
		 }
	      }	
	   }
	}
}

//Send Packet of 8008 Bytes each 
void* send_packet(void *td)
{
	struct send_packet_data* ThreadData = (struct send_packet_data*) td;
    	char *packet = (char *)malloc(sizeof(char)*packet_size);
   	int k, rv;
    	unsigned long long lMCNT=0;
    	double byte_count=0;

    	while(StartSendingPacket == 0){
       		if(run == 0){
          		exit(0);
       		}	
	} 

        while (run && (byte_count<total_data || total_data<=0)) {
	       for(k=0; k < ThreadData->Kmax; k++){
		   //header generation function call
		   header_gen(packet, lMCNT, ThreadData->m, 
                                    (unsigned char) k, ThreadData->cal);
		   //printf("%d %x\n", (int)ThreadData->m, *(packet+6));
		   //payload generation function call
		   payload_gen(packet+8, (unsigned char) k, ThreadData->SineCh, 
                                          ThreadData->SinePh, ThreadData->SinePer);

		   rv = sendto(sock[ThreadData->m], packet, (size_t)packet_size, 0, 
		               (struct sockaddr *)&ip_addr[k], slen);
		 
		   if (rv==-1) {
		       perror("sendto");
		       exit(1);
		   }
	 
		   int i;
		   for (i=0; i<(int)(1000.0*wait_cyc); i++) { __asm__("nop;nop;nop"); }
	       }

	       if(ThreadData->UseGlobalMCNT){
	          MCNT++;
	          lMCNT = MCNT;
	       }
	       else{
	          lMCNT++;
	       }

	       byte_count += (double)packet_size;
    	}

   	free(packet);

}

void main(int argc, char *argv[])
{
	
   	int Kmax = Nxengines; //1; /*Maximum no. of Xengines used*/
   	int Mmax = Nfengines; //5; /*Maximum no. of Fengines used*/
   	unsigned char cal = 0x0; 
   	int m, hcnt;
   	int port_num = PORT_NUM;
   	int SineCh = 0; 
   	float SinePh = 0; 
   	int SinePer = 0; 
   	int rv;
	
   	signal(SIGINT, cc);

   	/* Cmd line */
   	static struct option long_opts[] = {
		{"help",   0, NULL, 'h'},
		{"port",   1, NULL, 'p'},
		{"packet-size",   1, NULL, 's'},
		{"total-data",    1, NULL, 'd'},
		{"quiet",  0, NULL, 'q'},
		{"wait",   1, NULL, 'w'},
		{"endian", 0, NULL, 'e'},
		{"sine_channel", 1, NULL, 'c'},
		{"sine_phase", 1, NULL, 'a'},
		{0,0,0,0}
   	};

   	int quiet=0;
  	int opt, opti;
   	while ((opt=getopt_long(argc,argv,"hp:s:d:qw:ec:a:",long_opts,&opti))!=-1) {
		switch (opt) {
		    case 'p':
			port_num = atoi(optarg);
			break;
		    case 's':
			packet_size = atoi(optarg);
			break;
		    case 'd':
			total_data = atof(optarg);
			break;
		    case 'q':
			quiet=1;
			break;
		    case 'w':
			wait_cyc = atof(optarg);
			break;
		    case 'c':
			SineCh = atoi(optarg);
			break;
		    case 'e':
			endian=1;
			break;
		    case 'a':
			SinePh = atof(optarg);
			break;
		    case 'h':
		    default:
			usage();
			exit(0);
			break;
		}
    	}

   	if(!quiet){
      		printf("Mmax = %d , Kmax = %d\n", Mmax, Kmax);
   	}


   /* Send packets */
   	if(!quiet){
      		printf("Sine Channel  = %d \n", SineCh);
     		printf("Sine Phase (rad)  = %f \n", SinePh);
      		printf("Sine Per   = %d \n", SinePer);
      		printf("cal   = %x \n", cal);
      		printf("total_data (in GB) = %f \n", total_data);
   	}

   	for (hcnt=0; hcnt<Kmax; hcnt++) { 
                /* Resolve hostname */
        	hh[hcnt] = gethostbyname(ipaddr[hcnt]);

        	if ((hh[hcnt])==NULL) {
            	herror("gethostbyname");
            	exit(1);
        	}

        	/* Set up recvr address */
        	ip_addr[hcnt].sin_family = AF_INET;
        	ip_addr[hcnt].sin_port = htons(port_num);
        	memcpy(&ip_addr[hcnt].sin_addr, hh[hcnt]->h_addr, sizeof(struct in_addr));

        	if(!quiet){
           		printf("k=%d ipaddr=%s\n", hcnt, inet_ntoa(ip_addr[hcnt].sin_addr));
        	}
   	}

   	slen = sizeof(ip_addr[0]);

	//Mmax Thread ids 
	pthread_t thread_send_packet[Mmax];
	struct send_packet_data data_thread[Mmax];


   	int RoachUsingGlobalMCNT = -1;

   	for (m=0; m<Mmax; m++) {
       		/* Create socket */
       		sock[m] = socket(PF_INET, SOCK_DGRAM, 0);
       		if (sock[m]==-1) {
	   		perror("socket");
	   		exit(1);
       		}
		//Data for each thread is set here
		data_thread[m].m = m; 
		data_thread[m].cal = cal; 
		data_thread[m].Kmax = Kmax; 
		data_thread[m].SineCh = SineCh;
		data_thread[m].SinePh = SinePh;
		data_thread[m].SinePer = SinePer; 
   	        if(m == RoachUsingGlobalMCNT){
		    data_thread[m].UseGlobalMCNT = 1;
                }
                else{
		    data_thread[m].UseGlobalMCNT = 0;
                }
		//Create Thread
		pthread_create(&thread_send_packet[m], NULL, (void*)send_packet, &data_thread[m]);
     
   	}

   	total_data *= 1024.0*1024.0*1024.0; /* change to bytes */
   	StartSendingPacket=1; 

	int thread_done=1, ftc;

	//Thread Join to Synchronize 
	for(ftc = 0; ftc < Mmax; ftc ++){
		thread_done = pthread_join(thread_send_packet[ftc], NULL);
		if(thread_done != 0){
                   printf("Abnormal exist of thread %d \n", ftc);
                }
	}
}
