# serial 1 xgpu_pfb.m4
AC_DEFUN([AX_CHECK_XGPU_PFB],
[AC_PREREQ([2.65])dnl
AC_ARG_WITH([xgpupfb],
            AC_HELP_STRING([--with-xgpupfb=DIR],
                           [Location of xGPU_pfb headers/libs (/usr/local)]),
            [XGPU_PFB_DIR="$withval"],
            [XGPU_PFB_DIR=/usr/local])

orig_LDFLAGS="${LDFLAGS}"
LDFLAGS="${orig_LDFLAGS} -L${XGPU_PFB_DIR}/lib"
AC_CHECK_LIB([xgpupfb], [xgpuInfo],
             # Found
             AC_SUBST(XGPU_PFB_LIBDIR,${XGPU_PFB_DIR}/lib),
             # Not found there, check XGPU_PFB_DIR
             AS_UNSET(ac_cv_lib_xgpu_pfb_xgpuInit)
             LDFLAGS="${orig_LDFLAGS} -L${XGPU_PFB_DIR}"
             AC_CHECK_LIB([XGPUPFB], [xgpuInit],
                          # Found
                          AC_SUBST(XGPU_PFB_LIBDIR,${XGPU_PFB_DIR}),
                          # Not found there, error
                          AC_MSG_ERROR([xGPU_pfb library not found])))
LDFLAGS="${orig_LDFLAGS}"

AC_CHECK_FILE([${XGPU_PFB_DIR}/include/xgpu.h],
              # Found
              AC_SUBST(XGPU_PFB_INCDIR,${XGPUDIR}/include),
              # Not found there, check XGPU_PFB_DIR
              AC_CHECK_FILE([${XGPU_PFB_DIR}/xgpu.h],
                            # Found
                            AC_SUBST(XGPU_PFB_INCDIR,${XGPU_PFB_DIR}),
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
AC_DEFUN([AX_CHECK_XGPU_PFB_INFO],
[AC_PREREQ([2.65])dnl
AX_CHECK_XGPU_PFB
AC_CHECK_FILE([${XGPUDIR}/bin/xgpuinfo_pfb],
              # Found
              AC_SUBST(XGPU_FRB_BINDIR,${XGPU_PFB_DIR}/bin),
              # Not found there, check XGPUDIR
              AC_CHECK_FILE([${XGPU_PFB_DIR}/xgpuinfo_pfb],
                            # Found
                            AC_SUBST(XGPU_PFB_BINDIR,${XGPU_PFB_DIR}),
                            # Not found there, error
                            AC_MSG_ERROR([xgpuinfo_pfb program not found])))

AC_DEFINE_UNQUOTED([XGPU_PFB_NSTATION],
                   [`${XGPU_BINDIR}/xgpuinfo_pfb | sed -n '/Number of stations: /{s/.*: //;p}'`],
                   [PFB Mode Number of stations == Ninputs/2])

AC_DEFINE_UNQUOTED([XGPU_PFB_NFREQUENCY],
                   [`${XGPU_BINDIR}/xgpuinfo_pfb | sed -n '/Number of frequencies: /{s/.*: //;p}'`],
                   [PFB Mode Number of frequency channels per xGPU instance])

AC_DEFINE_UNQUOTED([XGPU_PFB_NTIME],
                   [`${XGPU_BINDIR}/xgpuinfo_pfb | sed -n '/time samples per GPU integration: /{s/.*: //;p}'`],
                   [PFB Mode Number of time samples (i.e. spectra) per xGPU integration])
])
