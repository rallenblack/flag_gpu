# Green Bank Installation Instructions

## Installation
### Environment

`FLAG_GPU`: the root of your FLAG_GPU installation. For example: `/home/sandboxes/tchamber/flag_gpu_test`

`HASH_DIR`: the root of your hashpipe installation. For example: `$FLAG_GPU/hash`

### Hashpipe
	cd lib/hashpipe/src
	$ autoreconf -is

Hashpipe needs to know where it should be installed

	$ ./configure --prefix=$HASH_DIR
	$ make
	$ make install

### xGPU
To build xGPU you will need to tell it where `xGPU` is installed. You can do this by creating a local `Makefile` that will do this for you:

	$ cd $FLAG_GPU/lib/xGPU/src
	$ echo "CUDA_DIR ?= /opt/local/cuda" > Makefile.local

We need to tell `xGPU` where to place its binaries, but without a `configure` script we will need to do this via a `make install` flag:

	$ make install prefix=$HASH_DIR

### Beamformer Shared Libarary
	$ cd $FLAG_GPU/lib/beamformer/src
	$ echo "CUDA_DIR ?= /opt/local/cuda" > Makefile.local
	$ make install prefix=$HASH_DIR

### Beamformer Hashpipe Plugin
	$ ./configure --prefix=$HASH_DIR --with-hashpipe=$HASH_DIR --with-xgpu=$HASH_DIR --with-flagbeamformer=$HASH_DIR

## Execution
First, we need to get onto `west`:

	$ ssh west

`$HASH_DIR/bin` must be in your PATH prior to running hashpipe. Then:

	$ hashpipe -p flag_beamformer -o XID=0 -o INSTANCE=0 -o BINDHOST=px1-2.gb.nrao.edu -c 0 flag_net_thread -c 1 flag_transpose_thread -c 2 flag_beamform_thread -c 3 flag_beamsave_thread
