# serial 1 xgpu_frb.m4
AC_DEFUN([AX_CHECK_XGPU_FRB],
[AC_PREREQ([2.65])dnl
AC_ARG_WITH([xgpufrb],
            AC_HELP_STRING([--with-xgpufrb=DIR],
                           [Location of xGPU_frb headers/libs (/usr/local)]),
            [XGPU_FRB_DIR="$withval"],
            [XGPU_FRB_DIR=/usr/local])

orig_LDFLAGS="${LDFLAGS}"
LDFLAGS="${orig_LDFLAGS} -L${XGPU_FRB_DIR}/lib"
AC_CHECK_LIB([xgpufrb], [xgpuInfo],
             # Found
             AC_SUBST(XGPU_FRB_LIBDIR,${XGPU_FRB_DIR}/lib),
             # Not found there, check XGPU_FRB_DIR
             AS_UNSET(ac_cv_lib_xgpu_frb_xgpuInit)
             LDFLAGS="${orig_LDFLAGS} -L${XGPU_FRB_DIR}"
             AC_CHECK_LIB([XGPUFRB], [xgpuInit],
                          # Found
                          AC_SUBST(XGPU_FRB_LIBDIR,${XGPU_FRB_DIR}),
                          # Not found there, error
                          AC_MSG_ERROR([xGPU_frb library not found])))
LDFLAGS="${orig_LDFLAGS}"

AC_CHECK_FILE([${XGPU_FRB_DIR}/include/xgpu.h],
              # Found
              AC_SUBST(XGPU_FRB_INCDIR,${XGPUDIR}/include),
              # Not found there, check XGPU_FRB_DIR
              AC_CHECK_FILE([${XGPU_FRB_DIR}/xgpu.h],
                            # Found
                            AC_SUBST(XGPU_FRB_INCDIR,${XGPU_FRB_DIR}),
                            # Not found there, error
                            AC_MSG_ERROR([XGPU.h header file not found])))
])

dnl Calls AX_CHECK_XGPU and then checks for and uses xgpuinfo to define the
dnl following macros in config.h:
dnl
dnl   XGPU_NSTATION   - Number of dual-pol(!) stations per xGPU instance
dnl   XGPU_NFREQUENCY - Number of frequency channels per xGPU instance
dnl   XGPU_NTIME      - Number of time samples per freqency channel per xGPU
dnl                     instance
dnl
AC_DEFUN([AX_CHECK_XGPU_FRB_INFO],
[AC_PREREQ([2.65])dnl
AX_CHECK_XGPU_FRB
AC_CHECK_FILE([${XGPUDIR}/bin/xgpuinfo_frb],
              # Found
              AC_SUBST(XGPU_FRB_BINDIR,${XGPU_FRB_DIR}/bin),
              # Not found there, check XGPUDIR
              AC_CHECK_FILE([${XGPU_FRB_DIR}/xgpuinfo_frb],
                            # Found
                            AC_SUBST(XGPU_FRB_BINDIR,${XGPU_FRB_DIR}),
                            # Not found there, error
                            AC_MSG_ERROR([xgpuinfo_frb program not found])))

AC_DEFINE_UNQUOTED([XGPU_FRB_NSTATION],
                   [`${XGPU_BINDIR}/xgpuinfo_frb | sed -n '/Number of stations: /{s/.*: //;p}'`],
                   [FRB Mode Number of stations == Ninputs/2])

AC_DEFINE_UNQUOTED([XGPU_FRB_NFREQUENCY],
                   [`${XGPU_BINDIR}/xgpuinfo_frb | sed -n '/Number of frequencies: /{s/.*: //;p}'`],
                   [FRB Mode Number of frequency channels per xGPU instance])

AC_DEFINE_UNQUOTED([XGPU_FRB_NTIME],
                   [`${XGPU_BINDIR}/xgpuinfo_frb | sed -n '/time samples per GPU integration: /{s/.*: //;p}'`],
                   [FRB Mode Number of time samples (i.e. spectra) per xGPU integration])
])
