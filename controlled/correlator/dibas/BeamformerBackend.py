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
from datetime import datetime, timedelta

from VegasBackend import VegasBackend

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

        self.fits_writer_program = 'bfFitsWriter'
        self.start_fits_writer()

    def start(self, inSecs, durSecs):
        "Start a scan in inSecs for durSecs long"

        # our fake GPU simulator needs to know the start time of the scan
        # and it's duration, so we need to write it to status shared mem.
        print "PLAYER: Starting scan in %d seconds for %d seconds" % (inSecs, durSecs)
        def secs_2_dmjd(secs):
            dmjd = (secs/86400) + 40587
            return dmjd + ((secs % 86400)/86400.)

        inSecs = inSecs if inSecs is not None else 5
        durSecs = durSecs if durSecs is not None else 5

        # TBF: we've done our stuff w/ DMJD, but our start is a utc datetime obj
        now = time.time()
        startDMJD = secs_2_dmjd(int(now + inSecs))

        # NOTE: SCANLEN can also be set w/ player.set_param(scan_length=#)
        self.write_status(STRTDMJD=str(startDMJD),SCANLEN=str(durSecs))
        self.scan_length = durSecs

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

        super(BeamformerBackend, self).prepare()

        self.set_register(acc_len=self.acc_len)

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

        # Clear shared memory for this instance
        subprocess.call("hashpipe_clean_shmem -I 0", shell=True)

        hpc_program = self.mode.hpc_program
        if hpc_program is None:
            raise Exception("Configuration error: no field hpc_program specified in "
                            "MODE section of %s " % (self.current_mode))

        # TBF: remove the special handling of beamformer
        if hpc_program == 'beamformer':
            # cmd = 'taskset 0x0606 hashpipe -p fake_gpu -I %d -o BINDHOST=px1-2.gb.nrao.edu -o GPUDEV=0 -o XID=0 -c 3 fake_gpu_thread' % self.instance_id
            cmd = 'hashpipe -p flag_sim2 -I %d -o BINDHOST=10.16.96.239 -o BINDPORT=50001 -o GPUDEV=0 -o XID=0 -c 0 flag_net_thread -c 1 flag_transpose_thread -c 2 flag_fakecor_thread -c 3 flag_corsave_thread' % self.instance_id
             
            process_list = shlex.split(cmd)
        else:
            process_list = [hpc_program]

        if self.mode.hpc_program_flags and hpc_program != 'beamformer':
            process_list = process_list + self.mode.hpc_program_flags.split()

        print "process_list: ", process_list
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
        #fits_writer_program = "bfFitsWriter"

        #cmd = self.dibas_dir + '/exec/x86_64-linux/' + fits_writer_program
        #self.fits_writer_process = subprocess.Popen((sp_path, ), stdin=subprocess.PIPE)
        #cmd += " -i %d" % self.instance_id
        #cmd += " -m c" 
        cmd = "dummy_fits_writer"
        process_list = shlex.split(cmd)
        self.fits_writer_process = subprocess.Popen(process_list, stdin=subprocess.PIPE)
    
