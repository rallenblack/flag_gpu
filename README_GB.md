# Green Bank Installation Instructions

## Installation
### Environment

In order to mirror the directory structure we have here in Green Bank, I will be assuming the following directory structure:

	$FLAG_DIR
	|-- hash
	|   |-- bin
	|   |-- include
	|   |-- lib
	|   `-- share
	`-- repos
		`-- flag_gpu

For example, my `FLAG_DIR` is `/home/sandboxes/tchamber/flag_gpu_test`

Now let's set two more variable to make things easier later on:

	$ export HASH_DIR=$FLAG_DIR/hash
	$ export FLAG_GPU=$FLAG_DIR/repos/flag_gpu

### Hashpipe
	cd $FLAG_GPU/lib/hashpipe/src
	$ autoreconf -is

Hashpipe needs to know where it should be installed

	$ ./configure --prefix=$HASH_DIR
	$ make
	$ make install

### xGPU
To build xGPU you will need to tell it where `xGPU` is installed. You can do this by creating a local `Makefile` that will do this for you:

	$ cd $FLAG_DIR/lib/xGPU/src
	$ echo "CUDA_DIR ?= /opt/local/cuda" > Makefile.local

We need to tell `xGPU` where to place its binaries, but without a `configure` script we will need to do this via a `make install` flag:

	$ make install prefix=$HASH_DIR

### Beamformer Shared Libarary
	$ cd $FLAG_DIR/lib/beamformer/src
	$ echo "CUDA_DIR ?= /opt/local/cuda" > Makefile.local
	$ make install prefix=$HASH_DIR

### Beamformer Hashpipe Plugin
	$ ./configure --prefix=$HASH_DIR --with-hashpipe=$HASH_DIR --with-xgpu=$HASH_DIR --with-flagbeamformer=$HASH_DIR

## Execution
First, we need to get onto `west`:

	$ ssh west

`$HASH_DIR/bin` must be in your PATH prior to running hashpipe. Then:

	$ hashpipe -p flag_beamformer -o XID=0 -o INSTANCE=0 -o BINDHOST=px1-2.gb.nrao.edu -c 0 flag_net_thread -c 1 flag_transpose_thread -c 2 flag_beamform_thread -c 3 flag_beamsave_thread
