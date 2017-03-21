AC_DEFUN([AX_CHECK_PYTHON], [

# provide CUDA path
AC_ARG_WITH(python,
			[--with-python=PREFIX  Prefix of your CUDA installation],
			[python_prefix=$withval], [python_prefix="/usr/local"])

# Set prefix to default if only the --with-cuda was passed
if test "$python_prefix" == "yes"; then
	if test "$withval" == "yes"; then
		python_prefix="/usr/local/lib"
	fi
fi

# add CUDA to search dir for header and lib searches

ax_save_CFLAGS="${CFLAGS}"
ax_save_LDFLAGS="${LDFLAGS}"

AC_SUBST([PYTHON_INCDIR])
AC_SUBST([PYTHON_LIBDIR])

PYTHON_INCDIR="-I$python_prefix/include/python2.7"
CFLAGS="$PYTHON_INCDIR $CFLAGS"
PYTHON_LIBDIR="-L$python_prefix/lib"
LDFLAGS="$PYTHON_LIBDIR $LDFLAGS"


AC_CHECK_HEADER([Python.h], [], AC_MSG_FAILURE([Couldn't find Python.h]), [#include <Python.h>])
AC_CHECK_LIB([python2.7], [Py_Initialize], [], AC_MSG_FAILURE([Couldn't find libpython]))

# Return to original flags
CFLAGS=${ax_save_CFLAGS}
LDFLAGS=${ax_save_LDFLAGS}


])
