
# flag_gpu
CPU/GPU codes for real-time beamforming and correlations for the Focal L-Band Array for the GBT (FLAG) phased-array feed. To be used with the hashpipe thread management system.

Filesystem Layout
flag_gpu/src            -> Source code for the various threads used in the FLAG digital processor
flag_gpu/lib            -> Source codes for the various libraries used by the FLAG threads
flag_gpu/lib/hashpipe   -> Source code for the hashpipe system
flag_gpu/lib/xGPU       -> Source code for the GPU-enabled correlator
flag_gpu/lib/flagPower  -> Source code for the GPU-enabled total power calculator
flag_gpu/lib/beamformer -> Source code for the GPU-enabled real-time beamformer
flag_gpu/scripts        -> Relevant python scripts for connection with Dealer/Player
flag_gpu/utils          -> Helpful scripts/programs for testing the system and interpreting output data products


Installation Instructions

1. Open the "install" script
2. Modify the line "prefix=" to point to an installation directory (e.g. /usr/local/bin)
3. Save and exit
4. Run the install script using "./install"
5. Append to your LD_LIBRARY_PATH environment variable, '<prefix>/lib'
6. Append to your PATH environment variables, '<prefix>/bin'


Running the Code

$ hashpipe -p <plug_in_name>


