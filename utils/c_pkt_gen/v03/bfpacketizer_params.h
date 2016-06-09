#define PACKET_SIZE 8008 		// Packet Size in Bytes
#define PORT_NUM 60000			// Port Number for UDP Server		
#define Ninputs 40			// Number of analog inputs
#define Nfengines 5			// Number of Roach-II boards used as F-engines
#define Nxengines 20			// Number of GPUs used for correlations
#define Ntime_per_packet 20		// Number of time stamps in a packet 
#define Nbins 500			// Number of freq channels used for correlations 
#define Nfft 512			// Number of PFB freq channels 
#define Nin_per_f() Ninputs/Nfengines;	// Number of analog inputs to each Roach-II board 
#define Nbin_per_x() Nbins/Nxengines;   // Number of freq channels processed in a GPU	

// Ip address of the GPUs 
const char *ipaddr[] = {"10.10.1.13",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1",
                        "10.40.5.1"};
