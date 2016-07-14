#serial 1 flagbeamform.m4
AC_DEFUN([AX_CHECK_FLAGBEAMFORM],
[AC_PREREQ([2.63])dnl
AC_ARG_WITH([flagbeamformer],
            AC_HELP_STRING([--with-flagbeamformer=DIR],
                           [Location of flagbeamformer headers/libs (/usr/local)]),
            [FLAGBEAMDIR="$withval"],
            [FLAGBEAMDIR=/usr/local])

orig_LDFLAGS="${LDFLAGS}"
LDFLAGS="${orig_LDFLAGS} -L${FLAGBEAMDIR}/lib -L/usr/local/cuda/lib64"
AC_CHECK_LIB([flagbeamformer], [update_weights],
dnl              # Found
             AC_SUBST(FLAGBEAM_LIBDIR,${FLAGBEAMDIR}/lib),
dnl              # Not found there, check FLAGBEAMDIR
             AS_UNSET(ac_cv_lib_flagbeamformer_run_beamformer)
             LDFLAGS="${orig_LDFLAGS} -L${FLAGBEAMDIR}"
             AC_CHECK_LIB([flagbeamformer], [update_weights],
                          # Found
                          AC_SUBST(FLAGBEAM_LIBDIR,${FLAGBEAMDIR}),
                          # Not found there, error
                          AC_MSG_ERROR([flagbeamformer library not found]), -lcublas), -lcublas)
LDFLAGS="${orig_LDFLAGS}"

AC_CHECK_FILE([${FLAGBEAMDIR}/include/cublas_beamformer.h],
              # Found
              AC_SUBST(FLAGPOW_INCDIR,${FLAGBEAMDIR}/include),
              # Not found there, check FLAGBEAMDIR
              AC_CHECK_FILE([${FLAGBEAMDIR}/cublas_beamformer.h],
                            # Found
                            AC_SUBST(FLAGBEAM_INCDIR,${FLAGBEAMDIR}),
                            # Not found there, error
                            AC_MSG_ERROR([cublas_beamformer.h header file not found])))
])
