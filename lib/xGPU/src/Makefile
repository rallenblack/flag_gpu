# Allow local customizations to build process
# (usually sizing for a specific project)
-include Makefile.local

CUDA_DIR ?= /usr/local/cuda
ifneq ($(strip $(OSTYPE)),osx)
CUDA_LIBDIR ?= $(CUDA_DIR)/lib64
else
CUDA_LIBDIR ?= $(CUDA_DIR)/lib
endif

DEBUG ?= -O3
TEXTURE_DIM ?= 1
CUDA_ARCH ?= sm_20

NVCC = $(CUDA_DIR)/bin/nvcc
prefix ?= /usr/local
bindir ?= $(prefix)/bin
libdir ?= $(prefix)/lib
includedir ?= $(prefix)/include

ifeq ($(strip $(OSTYPE)),cygwin)
NVCCFLAGS = --compiler-bindir "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin" -Xcompiler /openmp
LFLAGS = -L"c:\Program Files\Microsoft SDKs\Windows\v7.1\Lib" -L.
RPATH =
LIBXGPU = libxgpu.dll
LIBXGPU_FRB = libxgpufrb.dll
LIBXGPU_PFB = libxgpupfb.dll
else

NVCCFLAGS = -Xcompiler -D_REENTRANT -Xcompiler -Wall -Xcompiler -Wno-unused-function

ifneq ($(strip $(OSTYPE)),osx)
NVCCFLAGS += -Xcompiler -fopenmp 
else
NVCCFLAGS += -Xcompiler -Wno-unknown-pragmas -Xcompiler -Wno-gnu-designator
endif

ifneq ($(strip $(OSTYPE)),osx)
LFLAGS = -L$(CUDA_LIBDIR) -L. -lrt
else
LFLAGS = -L$(CUDA_LIBDIR) -L. -lcudart
endif

RPATH = -Xlinker -rpath,'$${ORIGIN}:$${ORIGIN}/../lib'
LIBXGPU = libxgpu.so
LIBXGPU_FRB = libxgpufrb.so
LIBXGPU_PFB = libxgpupfb.so
endif

PFB_FLAGS=
FRB_FLAGS =
X_FLAGS =

NVCCFLAGS += $(DEBUG) -arch=$(CUDA_ARCH) -Xptxas -abi=no --ptxas-options=-v -prec-sqrt=false -Xcompiler -fPIC 
NVCCFLAGS += -DTEXTURE_DIM=$(TEXTURE_DIM)

ifeq ($(strip $(CUDA_ARCH)),sm_35)
NVCCFLAGS += -maxrregcount=72
endif

ifeq ($(strip $(CUDA_ARCH)),sm_50)
NVCCFLAGS += -maxrregcount=72
endif

#NVCCFLAGS += -Xptxas -dlcm=cg # disable L1 cache

# Possibly (re-)build xgpu_version.h
$(shell $(SHELL) gen_xgpu_version.sh)

##########################
# Sizing variable handlers
##########################

# Force the output format to be triangular always! (RB 2/17)
NVCCFLAGS += -DMATRIX_ORDER_TRIANGULAR

ifdef NPOL
NVCCFLAGS += -DNPOL=$(NPOL)
endif

ifdef NSTATION
NVCCFLAGS += -DNSTATION=$(NSTATION)
else
# Handle common typo
ifdef NSTATIONS
NVCCFLAGS += -DNSTATION=$(NSTATIONS)
endif
endif

ifdef NFREQUENCY
NVCCFLAGS += -DNFREQUENCY=$(NFREQUENCY)
endif

ifdef NTIME
NVCCFLAGS += -DNTIME=$(NTIME)
endif

ifdef NTIME_PIPE
NVCCFLAGS += -DNTIME_PIPE=$(NTIME_PIPE)
else
# Handle common typo
ifdef NTIMEPIPE
NVCCFLAGS += -DNTIME_PIPE=$(NTIMEPIPE)
endif
endif

ifdef SHARED_ATOMIC_SIZE
NVCCFLAGS += -DSHARED_ATOMIC_SIZE=$(SHARED_ATOMIC_SIZE)
endif

ifdef COMPLEX_BLOCK_SIZE
NVCCFLAGS += -DCOMPLEX_BLOCK_SIZE=$(COMPLEX_BLOCK_SIZE)
endif

ifdef REAL_IMAG_TRIANGULAR_ORDER
NVCCFLAGS += -DMATRIX_ORDER_REAL_IMAG
endif

################################
# Benchmarking variable handlers
################################

ifdef CUBE_COUNT_MODE
NVCCFLAGS += -DCUBE_COUNT_MODE
CUBE_OBJS = cube/cube.o
endif

ifdef CUBE_TIME_MODE
# FIXME? The test seems to fail on the timing run - why is this? - summation is doubled for some reason
NVCCFLAGS += -DCUBE_TIME_MODE
CUBE_OBJS = cube/cube.o
endif

ifdef CUBE_ASYNC_COUNT_MODE
NVCCFLAGS += -DCUBE_ASYNC_COUNT_MODE
CUBE_OBJS = cube/cube.o
endif

ifdef CUBE_ASYNC_TIME_MODE
NVCCFLAGS += -DCUBE_ASYNC_TIME_MODE
CUBE_OBJS = cube/cube.o
endif

# this sets the pipeline to run in an infinite loop - for power measurement
ifeq ($(strip $(POWER_LOOP)), yes)
NVCCFLAGS += -DPOWER_LOOP
endif

ifdef RUNTIME_STATS
NVCCFLAGS += -DRUNTIME_STATS
endif

# Handle V=1 for verbose output
ifneq "$V" "1"
	VERBOSE=@
endif

CUDA_CORRELATOR_OBJS  = cuda_correlator.o

XGPUINFO_OBJS  = xgpuinfo.o
XGPUINFO_FRB_OBJS  = xgpuinfo_frb.o
XGPUINFO_PFB_OBJS = xgpuinfo_pfb.o

# Objects for executables
BINXGPU_OBJS = $(CUDA_CORRELATOR_OBJS) $(XGPUINFO_OBJS)
BINXGPU_FRB_OBJS = $(CUDA_CORRELATOR_OBJS) $(XGPUINFO_FRB_OBJS)
BINXGPU_PFB_OBJS = $(CUDA_CORRELATOR_OBJS) $(XGPUINFO_PFB_OBJS)

# Objects for library
LIBXGPU_OBJS  = cuda_xengine.o
LIBXGPU_OBJS += omp_xengine.o
LIBXGPU_OBJS += cpu_util.o
LIBXGPU_OBJS += $(CUBE_OBJS)

# Objects for frb library
LIBXGPU_FRB_OBJS  = cuda_xengine_frb.o
LIBXGPU_FRB_OBJS += omp_xengine_frb.o
LIBXGPU_FRB_OBJS += cpu_util_frb.o
LIBXGPU_FRB_OBJS += $(CUBE_OBJS)

# Objects for pfb library
LIBXGPU_PFB_OBJS  = cuda_xengine_pfb.o
LIBXGPU_PFB_OBJS += omp_xengine_pfb.o
LIBXGPU_PFB_OBJS += cpu_util_pfb.o
LIBXGPU_PFB_OBJS += $(CUBE_OBJS)

# Each object file has a corresponding dependency file
DEPS = $(BINXGPU_OBJS:.o=.d) $(LIBXGPU_OBJS:.o=.d)
FRB_DEPS = $(BINXGPU_FRB_OBJS:.o=.d) $(LIBXGPU_FRB_OBJS:.o=.d)
PFB_DEPS = $(BINXGPU_PFB_OBJS:.o=.d) $(LIBXGPU_PFB_OBJS:.o=.d)


all: cuda_correlator cuda_correlator_frb cuda_correlator_pfb xgpuinfo xgpuinfo_frb xgpuinfo_pfb $(LIBXGPU) $(LIBXGPU_FRB) $(LIBXGPU_PFB)

cuda_correlator: $(CUDA_CORRELATOR_OBJS) $(LIBXGPU)
	$(NVCC) $(NVCCFLAGS) $(CUDA_CORRELATOR_OBJS) $(X_FLAGS) -o $@ $(LFLAGS) $(RPATH) -lxgpu

cuda_correlator_frb: $(CUDA_CORRELATOR_OBJS) $(LIBXGPU_FRB)
	$(NVCC) $(NVCCFLAGS) $(CUDA_CORRELATOR_OBJS) $(FRB_FLAGS) -o $@ $(LFLAGS) $(RPATH) -lxgpufrb

cuda_correlator_pfb: $(CUDA_CORRELATOR_OBJS) $(LIBXGPU_PFB)
	$(NVCC) $(NVCCFLAGS) $(CUDA_CORRELATOR_OBJS) $(PFB_FLAGS) -o $@ $(LFLAGS) $(RPATH) -lxgpupfb

xgpuinfo: $(XGPUINFO_OBJS) $(LIBXGPU)
	$(NVCC) $(NVCCFLAGS) $(XGPUINFO_OBJS) $(X_FLAGS) -o $@ $(LFLAGS) $(RPATH) -lxgpu

xgpuinfo_frb: $(XGPUINFO_FRB_OBJS) $(LIBXGPU_FRB)
	$(NVCC) $(NVCCFLAGS) $(XGPUINFO_FRB_OBJS) $(FRB_FLAGS) -o $@ $(LFLAGS) $(RPATH) -lxgpufrb

xgpuinfo_pfb: $(XGPUINFO_PFB_OBJS) $(LIBXGPU_PFB)
	$(NVCC) $(NVCCFLAGS) $(XGPUINFO_PFB_OBJS) $(PFB_FLAGS) -o $@ $(LFLAGS) $(RPATH) -lxgpupfb

$(LIBXGPU): $(LIBXGPU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(X_FLAGS) $^ -o $@ $(LFLAGS) --shared

$(LIBXGPU_FRB): $(LIBXGPU_FRB_OBJS)
	$(NVCC) $(NVCCFLAGS) $(FRB_FLAGS) $^ -o $@ $(LFLAGS) --shared

$(LIBXGPU_PFB): $(LIBXGPU_PFB_OBJS)
	$(NVCC) $(NVCCFLAGS) $(PFB_FLAGS) $^ -o $@ $(LFLAGS) --shared

# A target to make all object files, but not link them
objs: $(BINXGPU_OBJS) $(LIBXGPU_OBJS) $(LIBXGPU_FRB_OBJS) $(LIBXGPU_PFB_OBJS)

# Include dependencies after explicit rules (really anywhere after first
# explicit rule) so that they do not provide default target.
-include $(DEPS) $(FRB_DEPS) $(PFB_DEPS)

# These .{c,cc,cu} -> .o rules will generate dependency files as a side effect
# of the compilation.  These files are included elsewhere in this Makefile.
#
# NB Include dep files using "-include", see
#    http://make.paulandlesley.org/autodep.html#traditional for more info.

%.o: %.c
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) -M -o $*.P $< && \
	cp $*.P $*.d && \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		  -e '/^$$/ d' -e 's/$$/ :/' < $*.P >> $*.d && \
	rm -f $*.P
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) -M -o $*.P $< && \
	cp $*.P $*.d && \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		  -e '/^$$/ d' -e 's/$$/ :/' < $*.P >> $*.d && \
	rm -f $*.P
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) -c -o $@ $<

install: xgpuinfo $(LIBXGPU) xgpuinfo_frb $(LIBXGPU_FRB) xgpuinfo_pfb $(LIBXGPU_PFB)
	mkdir -p $(bindir)
	cp xgpuinfo $(bindir)
	cp xgpuinfo_frb $(bindir)
	cp xgpuinfo_pfb $(bindir)
	mkdir -p $(includedir)
	cp xgpu.h $(includedir)
	cp xgpu_pfb.h $(includedir)
	mkdir -p $(libdir)
	cp $(LIBXGPU) $(libdir)
	cp $(LIBXGPU_FRB) $(libdir)
	cp $(LIBXGPU_PFB) $(libdir)

uninstall:
	rm -f $(bindir)/xgpuinfo
	rm -f $(includedir)/xgpu.h
	rm -f $(libdir)/$(LIBXGPU)
	rm -f $(bindir)/xgpuinfo_frb
	rm -f $(libdir)/$(LIBXGPU_FRB)
	rm -f $(bindir)/xgpuinfo_pfb
	rm -f $(libdir)/$(LIBXGPU_PFB)

tags:
	ctags -R .

show:
	@echo NINPUTS=`expr 2 '*' $(NSTATION)`
	@echo NSTATION=$(NSTATION)
	@echo NFREQUENCY=$(NFREQUENCY)
	@echo NTIME=$(NTIME)
	@echo NTIME_PIPE=$(NTIME_PIPE)
	@echo SHARED_ATOMIC_SIZE=$(SHARED_ATOMIC_SIZE)
	@echo COMPLEX_BLOCK_SIZE=$(COMPLEX_BLOCK_SIZE)

clean:
	rm -f cuda_correlator cuda_correlator_frb cuda_correlator_pfb libxgpu.so xgpuinfo xgpuinfo_frb xgpuinfo_pfb libxgpufrb.so libxgpupfb.so
	rm -f $(BINXGPU_FRB_OBJS) $(BINXGPU_OBJS) $(BINXGPU_PFB_OBJS) $(LIBXGPU_OBJS) $(LIBXGPU_FRB_OBJS) $(LIBXGPU_PFB_OBJS) cube/cube.o
	rm -f $(DEPS) $(FRB_DEPS) $(PFB_DEPS)
	rm -f tags

.PHONY : all objs tags clean install uninstall show
