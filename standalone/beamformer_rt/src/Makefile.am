ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS     =

AM_CPPFLAGS += -I"@HASHPIPE_INCDIR@"
AM_CPPFLAGS += -I"@XGPU_INCDIR@"
AM_CPPFLAGS += -I"@FLAGBEAM_INCDIR@"

# AM_CFLAGS is used for all C compiles
# AM_CFLAGS = -ggdb -fPIC -O3 -Wall -Werror -fno-strict-aliasing -mavx
AM_CFLAGS = -ggdb -fPIC -O3 -Wall -Werror -fno-strict-aliasing

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

flag_databuf = flag_databuf.h \
               flag_databuf.c

flag_threads = flag_net_thread.c \
               flag_transpose_thread.c \
               flag_beamform_thread.c \
               flag_beamsave_thread.c

# This is the flag_gpu plugin itself
lib_LTLIBRARIES = flag_beamformer.la
flag_beamformer_la_SOURCES  = $(flag_databuf) $(flag_threads)
flag_beamformer_la_LIBADD   = -lrt -lxgpu -lflagbeamformer -lcublas -L/usr/local/cuda/lib64
flag_beamformer_la_LDFLAGS  = -avoid-version -module -shared -export-dynamic --enable-shared -ansi
flag_beamformer_la_LDFLAGS += -L"@XGPU_LIBDIR@" -Wl,-rpath,"@XGPU_LIBDIR@"
flag_beamformer_la_LDFLAGS += -L"@HASHPIPE_LIBDIR@" -Wl,-rpath,"@HASHPIPE_LIBDIR@"
flag_beamformer_la_LDFLAGS += -L"@FLAGBEAM_LIBDIR@" -Wl, -rpath, "@FLAGBEAM_LIBDIR@"

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
