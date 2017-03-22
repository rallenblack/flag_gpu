ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS     =

AM_CPPFLAGS += -I"@HASHPIPE_INCDIR@"
AM_CPPFLAGS += -I"@XGPU_INCDIR@"
AM_CPPFLAGS += -I"@FLAGBEAM_INCDIR@"
AM_CPPFLAGS += -I"@FLAGPFB_INCDIR@"
AM_CPPFLAGS += -I"@FLAGPOW_INCDIR@"
AM_CPPFLAGS += -I"@XGPU_FRB_INCDIR@"

AM_CPPFLAGS += "@CUDA_INCDIR@"

# AM_CFLAGS is used for all C compiles
AM_CFLAGS = -fPIC -O3 -Wall -Werror -fno-strict-aliasing -mavx
# AM_CFLAGS = -ggdb -fPIC -O3 -Wall -Werror -fno-strict-aliasing

# Convenience variables to group source files
#paper_databuf = paper_databuf.h \
#	        paper_databuf.c

#paper_threads = paper_fake_net_thread.c       \
#		paper_gpu_cpu_output_thread.c \
#		paper_gpu_output_thread.c     \
#		paper_gpu_thread.c            \
#		paper_fluff_thread.c          \
#		paper_net_thread.c            \
#		paper_crc_thread.c

#paper_fluff = paper_fluff.h \
#	      paper_fluff.c

fifo_codes = fifo.h \
             fifo.c

flag_databuf = flag_databuf.h \
               flag_databuf.c

flag_x_threads = flag_net_thread.c \
                 flag_transpose_thread.c \
                 flag_correlator_thread.c \
                 flag_corsave_thread.c

flag_b_threads = flag_net_thread.c \
                 flag_transpose_thread.c \
                 flag_beamform_thread.c \
                 flag_beamsave_thread.c

flag_f_threads = flag_net_thread.c \
                 flag_transpose_thread.c \
                 flag_pfb_thread.c

flag_bx_threads = flag_net_thread.c \
                  flag_transpose_thread.c \
                  flag_dual_thread.c \
                  flag_dualsave_thread.c

flag_x_frb_threads = flag_net_thread.c \
                     flag_frb_transpose_thread.c \
                     flag_frb_correlator_thread.c \
                     flag_frb_corsave_thread.c

# This lists all of the plugins that will be created
lib_LTLIBRARIES = flag_x.la flag_b.la flag_f.la flag_x_frb.la flag_bx.la

# flag_x.la sources and libraries
flag_x_la_SOURCES  = $(flag_databuf) $(flag_x_threads) $(fifo_codes)
flag_x_la_LIBADD   = -lrt -lxgpu -L/usr/local/cuda/lib64 -lcfitsio
flag_x_la_LDFLAGS  = -avoid-version -module -shared -export-dynamic --enable-shared
flag_x_la_LDFLAGS += -L"@XGPU_LIBDIR@" -Wl,-rpath,"@XGPU_LIBDIR@"
flag_x_la_LDFLAGS += -L"@HASHPIPE_LIBDIR@" -Wl,-rpath,"@HASHPIPE_LIBDIR@"

# flag_b.la sources and libraries
flag_b_la_SOURCES  = $(flag_databuf) $(flag_b_threads) $(fifo_codes)
flag_b_la_LIBADD   = -lrt -lxgpu -lflagbeamformer -lcublas -L/usr/local/cuda/lib64 -lcfitsio
flag_b_la_LDFLAGS  = -avoid-version -module -shared -export-dynamic --enable-shared
flag_b_la_LDFLAGS += -L"@XGPU_LIBDIR@" -Wl,-rpath,"@XGPU_LIBDIR@"
flag_b_la_LDFLAGS += -L"@HASHPIPE_LIBDIR@" -Wl,-rpath,"@HASHPIPE_LIBDIR@"
flag_b_la_LDFLAGS += -L"@FLAGBEAM_LIBDIR@" -Wl, -rpath, "@FLAGBEAM_LIBDIR@"

# flab_f.la sources and libraries
flag_f_la_SOURCES = $(flag_databuf) $(flag_f_threads) $(fifo_codes)
flag_f_la_LIBADD  = -lrt -ldl -lutil -L/usr/local/lib -lpython2.7 -lxgpu -lflagpfb -lcufft -lcudart -L/usr/local/cuda/lib64 -lcfitsio
flag_f_la_LDFLAGS = -avoid-version -module -shared -export-dynamic --enable-shared
flag_f_la_LDFLAGS += -L"@XGPU_LIBDIR@" -Wl, -rpath, "@XGPU_LIBDIR@"
flag_f_la_LDFLAGS += -L"@HASHPIPE_LIBDIR@" -Wl, -rpath,"@HASHPIPE_LIBDIR@"
flag_f_la_LDFLAGS += -L"@FLAGPFB_LIBDIR@" -Wl, -rpath,"@FLAGPFB_LIBDIR@"

# flag_bx.la sources and libraries
flag_bx_la_SOURCES = $(flag_databuf) $(flag_bx_threads) $(fifo_codes)
flag_bx_la_LIBADD   = -lrt -lxgpu -lflagbeamformer -lcublas -L/usr/local/cuda/lib64 -lcfitsio
flag_bx_la_LDFLAGS  = -avoid-version -module -shared -export-dynamic --enable-shared
flag_bx_la_LDFLAGS += -L"@XGPU_LIBDIR@" -Wl,-rpath,"@XGPU_LIBDIR@"
flag_bx_la_LDFLAGS += -L"@HASHPIPE_LIBDIR@" -Wl,-rpath,"@HASHPIPE_LIBDIR@"
flag_bx_la_LDFLAGS += -L"@FLAGBEAM_LIBDIR@" -Wl, -rpath, "@FLAGBEAM_LIBDIR@"

# flag_x_frb.la sources and libraries
flag_x_frb_la_SOURCES = $(flag_databuf) $(flag_x_frb_threads) $(fifo_codes)
flag_x_frb_la_LIBADD   = -lrt -lxgpufrb -lflagbeamformer -lcublas -L/usr/local/cuda/lib64 -lcfitsio
flag_x_frb_la_LDFLAGS  = -avoid-version -module -shared -export-dynamic --enable-shared
flag_x_frb_la_LDFLAGS += -L"@XGPU_FRB_LIBDIR@" -Wl,-rpath,"@XGPU_FRB_LIBDIR@"
flag_x_frb_la_LDFLAGS += -L"@HASHPIPE_LIBDIR@" -Wl,-rpath,"@HASHPIPE_LIBDIR@"

# Installed scripts
# dist_bin_SCRIPTS = paper_init.sh \
#		   paper_crc_init.sh \
#		   xtor_down.sh \
#		   xtor_up.sh

# Non-installed test programs
#noinst_PROGRAMS = paper_fluff_test
#paper_fluff_test_SOURCES = paper_fluff_test.c $(paper_fluff)
# The paper_fluff_test_CFLAGS is an autotools workaround.
# http://www.gnu.org/software/automake/manual/html_node/Objects-created-both-with-libtool-and-without.html
#paper_fluff_test_CFLAGS = $(AM_CFLAGS)
#paper_fluff_test_LDADD = -lrt

# vi: set ts=8 noet :
