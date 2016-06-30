######################################################################
#
#  BeamformerBackend.py -- Instance class for the DIBAS Vegas
#  modes. Derives Vegas mode functionality from VegasBackend, adding
#  only HBW specific functionality, and I/O with the roach and with the
#  status shared memory.
#
#  Copyright (C) 2014 Associated Universities, Inc. Washington DC, USA.
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#  General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
#
#  Correspondence concerning GBT software should be addressed as follows:
#  GBT Operations
#  National Radio Astronomy Observatory
#  P. O. Box 2
#  Green Bank, WV 24944-0002 USA
#
######################################################################

import time
import shlex
import subprocess
import os
from datetime import datetime, timedelta

from VegasBackend import VegasBackend
import ConfigParser # We have additional parameters that are in the config file
from ConfigData import _ip_string_to_int

class BeamformerBackend(VegasBackend):
    """A class which implements the FLAG Beamformer functionality. 
    and which communicates with the roach and with the HPC programs via
    shared memory.

    BeamformerBackend(theBank, theMode, theRoach = None, theValon = None)

    Where:

    * *theBank:* Instance of specific bank configuration data BankData.
    * *theMode:* Instance of specific mode configuration data ModeData.
    * *theRoach:* Instance of katcp_wrapper
    * *theValon:* instance of ValonKATCP
    * *unit_test:* Set to true to unit test. Will not attempt to talk to
      roach, shared memory, etc.

    """

    def __init__(self, theBank, theMode, theRoach, theValon, hpc_macs, unit_test = False, instance_id = None):
        """
        Creates an instance of BeamformerBackend
        """

        VegasBackend.__init__(self, theBank, theMode, theRoach, theValon, hpc_macs, unit_test)

        # Create the ARP table from the mac addresses in dibas.conf under [HPCMACS]
        self.hpc_macs = hpc_macs
        if self.roach:
            self.table = []
            for m in range(256):
                self.table.append(0xFFFFFFFFFFFF)
            for mac in self.hpc_macs:
                self.table[mac] = self.hpc_macs[mac]

        # Add additional dealer-controlled parameters
        self.params["int_length"] = self.setIntegrationTime

        # Read in the additional parameters from the configuration file
        self.read_parameters(theBank, theMode)

        # Generate ROACH 10-GbE source mac addresses
        self.mac_base0 = 0x020200000000 + (2**8)*self.xid + 1
        self.mac_base1 = 0x020200000000 + (2**8)*self.xid + 2
        self.mac_base2 = 0x020200000000 + (2**8)*self.xid + 3
        self.mac_base3 = 0x020200000000 + (2**8)*self.xid + 4

        self.source_ip0 = 10*(2**24) + 10*(2**16) + 1*(2**8) + 101 + self.xid*4
        self.source_ip1 = 10*(2**24) + 10*(2**16) + 1*(2**8) + 102 + self.xid*4
        self.source_ip2 = 10*(2**24) + 10*(2**16) + 1*(2**8) + 103 + self.xid*4
        self.source_ip3 = 10*(2**24) + 10*(2**16) + 1*(2**8) + 104 + self.xid*4

        # TODO: Add source port to configuration file
        self.source_port = 60000

        # In the Beamformer, there are 2 GPUs, so we run multiple instances
        # of the 'pipeline': more then one player!
        # should be set in base class ...
        if instance_id is not None:
            self.instance_id = instance_id

        self.progdev()
        self.net_config()

        if self.mode.roach_kvpairs:
            self.write_registers(**self.mode.roach_kvpairs)

        self.reset_roach()
        self.prepare()
        self.clear_switching_states()
        self.add_switching_state(1.0, blank = False, cal = False, sig_ref_1 = False)
        self.start_hpc()

        # self.fits_writer_program = 'bfFitsWriter'
        self.start_fits_writer()




    def read_parameters(self, theBank, theMode):
        # Quick little process to convert IP addresses from raw integer values
        int2ip  = lambda n: '.'.join([str(n >> (i << 3) & 0xFF) for i in range(0,4)[::-1]])

        bank = theBank.name
        mode = theMode.name
        dibas_dir = os.getenv('DIBAS_DIR') # Should always succeed since player started up

        # Create a quick ConfigParser to parse the extra information in dibas.conf
        config = ConfigParser.ConfigParser()
        filename = dibas_dir + '/etc/config/dibas.conf'
        config.readfp(open(filename))

        # Extract the XID
        self.xid = config.getint(bank, 'xid')

        # Extract the Hashpipe instance number
        self.instance = config.getint(bank, 'instance')

        # Extract the GPU device index
        self.gpudev = config.getint(bank, 'gpudev')

        # Extract the list of cpu cores on which to run the threads
        self.cpus = config.get(bank, 'cpus')
        self.core = [int(x) for x in self.cpus.split(',') if x.strip().isdigit()]

        # Get the 10 GbE BINDHOST and BINDPORT for this player
        self.bindhost = int2ip(theBank.dest_ip)
        self.bindport = theBank.dest_port

        # Get the destination IP addresses
        macs = config.items('HPCMACS')
        idx = 0
        self.dest_ip = {}
        for i in macs:
            self.dest_ip[idx] = _ip_string_to_int(i[0])
            idx = idx + 1

        # Get the FITS writer process name
        self.fits_writer_program = config.get(mode, 'fits_process')


    def start(self, inSecs, durSecs):
        "Start a scan in inSecs for durSecs long"

        # our fake GPU simulator needs to know the start time of the scan
        # and it's duration, so we need to write it to status shared mem.
        def secs_2_dmjd(secs):
            dmjd = (secs/86400) + 40587
            return dmjd + ((secs % 86400)/86400.)

        inSecs = inSecs if inSecs is not None else 5
        durSecs = durSecs if durSecs is not None else 5
        self.scan_length = durSecs

        # TBF: we've done our stuff w/ DMJD, but our start is a utc datetime obj
        now = time.time()
        startDMJD = secs_2_dmjd(int(now + inSecs))

        # NOTE: SCANLEN can also be set w/ player.set_param(scan_length=#)
        self.write_status(STRTDMJD=str(startDMJD),SCANLEN=str(durSecs))
        self.write_status(REQSTI=str(self.requested_integration_time))

        dt = datetime.utcnow()
        dt.replace(second = 0)
        dt.replace(microsecond = 0)
        dt += timedelta(seconds = inSecs)

        VegasBackend.start(self, starttime = dt)

        #
    # prepare() for this class calls the base class prepare then does
    # the bare minimum required just for this backend, and then writes
    # to hardware, HPC, shared memory, etc.
    def prepare(self):
        """
        This command writes calculated values to the hardware and status memory.
        This command should be run prior to the first scan to properly setup
        the hardware.

        The sequence of commands to set up a measurement is thus typically::

          be.set_param(...)
          be.set_param(...)
          ...
          be.set_param(...)
          be.prepare()
        """

        # super(BeamformerBackend, self).prepare()

        # self.set_register(acc_len=self.acc_len)

        # Talk to outside things: status memory, HPC programs, roach

        if self.bank is not None:
            self.write_status(**self.status_mem_local)
        else:
            for i in self.status_mem_local.keys():
                print "%s = %s" % (i, self.status_mem_local[i])

        if self.roach:
            self.write_registers(**self.roach_registers_local)


    def _sampler_frequency_dep(self):
        """
        Computes the effective frequency of the A/D sampler for HBW mode
        """
        self.sampler_frequency = self.frequency * 1e6 * 2

    # _set_state_table_keywords() overrides the parent version, but
    # should call the parent version first thing, as it will build on
    # what the parent function does. Since the parent class prepare()
    # calls this, no need to call this from this Backend's prepare.
    def _set_state_table_keywords(self):
        """
        Gather status sets here
        Not yet sure what to place here...
        """

        super(BeamformerBackend, self)._set_state_table_keywords()
        self.set_status(BW_MODE = "high")
        self.set_status(OBS_MODE = "HBW")

    def set_instance_id(self, instance_id):
        self.instance_id = instance_id

    def start_hpc(self):
        """
        Beamformer mode's are hashpipe plugins that require special handling:
           * are not in the dibas install area
           * need to pass on the instance id
        """

        if self.test_mode:
            return

        self.stop_hpc()

        # Get hashpipe command (specified by configuration file)
        hpc_program = self.mode.hpc_program
        if hpc_program is None:
            raise Exception("Configuration error: no field hpc_program specified in "
                            "MODE section of %s " % (self.current_mode))

        # Create command to start process
        process_list = [hpc_program]

        # Add flags specified by configuration file
        if self.mode.hpc_program_flags:
            process_list = process_list + self.mode.hpc_program_flags.split()

        # Add Instance ID
        inst_str = "-I " + str(self.instance)
        process_list = process_list + inst_str.split()

        # Add BINDHOST
        host_str = "-o BINDHOST=" + str(self.bindhost)
        process_list = process_list + host_str.split()

        # Add BINDPORT
        port_str = "-o BINDPORT=" + str(self.bindport)
        process_list = process_list + port_str.split()

        # Add XID
        xid_str = "-o XID=" + str(self.xid)
        process_list = process_list + xid_str.split()

        # Add GPUDEV
        gpu_str = "-o GPUDEV=" + str(self.gpudev)
        process_list = process_list + gpu_str.split()

        # Add threads
        thread1 = "-c %d flag_net_thread" % (self.core[0])
        thread2 = "-c %d flag_transpose_thread" % (self.core[1])
        thread3 = "-c %d flag_correlator_thread" % (self.core[2])
        # thread4 = "-c %d flag_corsave_thread" % (self.core[3])

        process_list = process_list + thread1.split()
        process_list = process_list + thread2.split()
        process_list = process_list + thread3.split()
        # process_list = process_list + "-c 3 flag_corsave_thread".split()

        print ' '.join(process_list)
        self.hpc_process = subprocess.Popen(process_list, stdin=subprocess.PIPE)

    def start_fits_writer(self):
        """
        start_fits_writer()
        Starts the fits writer program running. Stops any previously running instance.
        For the beamformer, we have to pass on the instance id.
        """

        if self.test_mode:
            return

        self.stop_fits_writer()
        #fits_writer_program = "dummy_fits_writer"
        #fits_writer_program = "bfFitsWriter"

        cmd = self.dibas_dir + '/exec/x86_64-linux/' + self.fits_writer_program
        #self.fits_writer_process = subprocess.Popen((sp_path, ), stdin=subprocess.PIPE)
        cmd += " -i %d" % self.instance_id
        cmd += " -m c" 
        process_list = shlex.split(cmd)
        self.fits_writer_process = subprocess.Popen(process_list, stdin=subprocess.PIPE)

    ######################################################
    # ROACH-specific functions
    # Added by Richard B. on June 21, 2016
    ######################################################
    
    def progdev(self, bof = None):
        """progdev(self, bof):
        
        Programs the ROACH2 with boffile 'bof.'

        *bof:*
          A string, the name of the bof file. This parameter defaults
          to 'None'; if no bof file is specified, the function will
          load the bof file specified for the current mode, which is
          specified in that mode's section of the configuration file.
          A 'KeyError' will result if the current mode is not set.

        Overrrides progdev in Backend.py since BeamformerBackend does
        not use Valon for ROACH clocking.
        """

        if not bof:
            bof = self.mode.bof

        if self.roach:
            reply, informs = self.roach._request("progdev", 20.0) # Second argument is timeout

            if reply.arguments[0] != 'ok':
                print "Warning! FPGA was not deprogrammed."

            print "progdev programming bof", str(bof)
            return self.roach.progdev(str(bof))
        

    def net_config(self, data_ip = None, data_port = None, dest_ip = None, dest_port = None):
        """net_config(self)
        
        Configures the 10Gb/s interfaces on the roach. All relevant
        parameters are specified in the configuration file.

        TBF: Flesh out the description more (Richard B.)
        """
        if self.roach:
            self.roach.listdev()

            # Write XIDs
            self.roach.write_int('part2_x_id0', 12)
            self.roach.write_int('part2_x_id1', 13)
            self.roach.write_int('part2_x_id2', 14)
            self.roach.write_int('part2_x_id3', 15)
            self.roach.write_int('part2_x_id4', 12)
            self.roach.write_int('part2_x_id5', 13)
            self.roach.write_int('part2_x_id6', 14)
            self.roach.write_int('part2_x_id7', 15)
            self.roach.write_int('part2_x_id8', 12)
            self.roach.write_int('part2_x_id9', 13)
            self.roach.write_int('part2_x_id10', 14)
            self.roach.write_int('part2_x_id11', 15)
            self.roach.write_int('part2_x_id12', 12)
            self.roach.write_int('part2_x_id13', 13)
            self.roach.write_int('part2_x_id14', 14)
            self.roach.write_int('part2_x_id15', 15)
            self.roach.write_int('part2_x_id16', 12)
            self.roach.write_int('part2_x_id17', 13)
            self.roach.write_int('part2_x_id18', 14)
            self.roach.write_int('part2_x_id19', 15)

            # Write fabric port
            self.roach.write_int('part2_x_port', self.bindport)

            # Write Destination IP addresses
            self.roach.write_int('part2_x_ip0', int(self.dest_ip[12]))
            self.roach.write_int('part2_x_ip1', int(self.dest_ip[13]))
            self.roach.write_int('part2_x_ip2', int(self.dest_ip[14]))
            self.roach.write_int('part2_x_ip3', int(self.dest_ip[15]))
            self.roach.write_int('part2_x_ip4', int(self.dest_ip[12]))
            self.roach.write_int('part2_x_ip5', int(self.dest_ip[13]))
            self.roach.write_int('part2_x_ip6', int(self.dest_ip[14]))
            self.roach.write_int('part2_x_ip7', int(self.dest_ip[15]))
            self.roach.write_int('part2_x_ip8', int(self.dest_ip[12]))
            self.roach.write_int('part2_x_ip9', int(self.dest_ip[13]))
            self.roach.write_int('part2_x_ip10', int(self.dest_ip[14]))
            self.roach.write_int('part2_x_ip11', int(self.dest_ip[15]))
            self.roach.write_int('part2_x_ip12', int(self.dest_ip[12]))
            self.roach.write_int('part2_x_ip13', int(self.dest_ip[13]))
            self.roach.write_int('part2_x_ip14', int(self.dest_ip[14]))
            self.roach.write_int('part2_x_ip15', int(self.dest_ip[15]))
            self.roach.write_int('part2_x_ip16', int(self.dest_ip[12]))
            self.roach.write_int('part2_x_ip17', int(self.dest_ip[13]))
            self.roach.write_int('part2_x_ip18', int(self.dest_ip[14]))
            self.roach.write_int('part2_x_ip19', int(self.dest_ip[15]))
            time.sleep(0.5)

            # Start the 10GbE cores and populate the ARP table
            self.roach.config_10gbe_core('part2_gbe0', self.mac_base0, self.source_ip0, self.source_port, self.table)
            self.roach.config_10gbe_core('part2_gbe1', self.mac_base1, self.source_ip1, self.source_port, self.table)
            self.roach.config_10gbe_core('part2_gbe2', self.mac_base2, self.source_ip2, self.source_port, self.table)
            self.roach.config_10gbe_core('part2_gbe3', self.mac_base3, self.source_ip3, self.source_port, self.table)
            time.sleep(0.5)

        







