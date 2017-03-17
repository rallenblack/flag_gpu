AC_DEFUN([AX_CHECK_CUDA], [

# provide CUDA path
AC_ARG_WITH(cuda,
			[--with-cuda=PREFIX  Prefix of your CUDA installation],
			[cuda_prefix=$withval], [cuda_prefix="/usr/local/cuda"])

# Set prefix to default if only the --with-cuda was passed
if test "$cuda_prefix" == "yes"; then
	if test "$withval" == "yes"; then
		cuda_prefix="/usr/local/cuda"
	fi
fi

# check for nvcc
AC_MSG_CHECKING([nvcc in $cuda_prefix/bin])
if test -x "$cuda_prefix/bin/nvcc"; then
	AC_MSG_RESULT([found])
	AC_DEFINE_UNQUOTED([NVCC_PATH], ["$cuda_prefix/bin/nvccc"], [Path to nvcc binary])
else
	AC_MSG_RESULT([not found!])
	AC_MSG_FAILURE([nvcc was not found in $cuda_prefix/bin])
fi

# add CUDA to search dir for header and lib searches

ax_save_CFLAGS="${CFLAGS}"
ax_save_LDFLAGS="${LDFLAGS}"

AC_SUBST([CUDA_CFLAGS])
AC_SUBST([CUDA_LDFLAGS])

CUDA_CFLAGS="-I$cuda_prefix/include"
CFLAGS="$CUDA_CFLAGS $CFLAGS"
CUDA_LDFLAGS="-L$cuda_prefix/lib"
LDFLAGS="$CUDA_LDFLAGS $LDFLAGS"


AC_CHECK_HEADER([cuda.h], [], AC_MSG_FAILURE([Couldn't find cuda.h]), [#include <cuda.h>])
AC_CHECK_LIB([cuda], [cuInit], [], AC_MSG_FAILURE([Couldn't, find libcuda]))

# Return to original flags?
CFLAGS=${ax_save_CFLAGS}
LDFLAGS=${ax_save_LDFLAGS}


])