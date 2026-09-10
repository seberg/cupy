# distutils: language = c++

import atexit
import functools
import os
import warnings

import cython
import numpy

import cupy
from cupy_backends.cuda.api cimport runtime


cython_build_ver = CUPY_CYTHON_VERSION
# Empty when not built with CUDA Python.
cuda_bindings_build_ver = CUPY_CUDA_BINDINGS_VERSION or None
nvmath_build_ver = CUPY_NVMATH_VERSION or None


ENABLE_SLICE_COPY = bool(
    int(os.environ.get('CUPY_EXPERIMENTAL_SLICE_COPY', 0)))


CUDA_ARRAY_INTERFACE_SYNC = bool(
    int(os.environ.get('CUPY_CUDA_ARRAY_INTERFACE_SYNC', 1)))
CUDA_ARRAY_INTERFACE_EXPORT_VERSION = int(
    os.environ.get('CUPY_CUDA_ARRAY_INTERFACE_EXPORT_VERSION', 3))


DLPACK_EXPORT_VERSION = tuple(
    [
        int(x)
        for x in os.environ.get('CUPY_DLPACK_EXPORT_VERSION', '0.6').split('.')
    ]
)


cdef list _memos = []


def memoize(bint for_each_device=False):
    """Makes a function memoizing the result for each argument and device.

    This decorator provides automatic memoization of the function result.

    Args:
        for_each_device (bool): If ``True``, it memoizes the results for each
            device. Otherwise, it memoizes the results only based on the
            arguments.

    """
    def decorator(f):
        memo = {}
        _memos.append(memo)

        @functools.wraps(f)
        @cython.binding(True)
        def ret(*args, **kwargs):
            cdef int id = -1
            cdef dict m = memo
            if for_each_device:
                id = runtime.getDevice()
            if len(kwargs):
                arg_key = (id, args, frozenset(kwargs.items()))
            else:
                arg_key = (id, args)
            result = m.get(arg_key, m)
            if result is m:
                result = f(*args, **kwargs)
                result = m.setdefault(arg_key, result)
            return result

        return ret

    return decorator


@atexit.register
def clear_memo():
    """Clears the memoized results for all functions decorated by memoize."""
    for memo in _memos:
        memo.clear()


def experimental(api_name):
    """Declares that user is using an experimental feature.

    The developer of an API can mark it as *experimental* by calling
    this function. When users call experimental APIs, :class:`FutureWarning`
    is issued.
    The presentation of :class:`FutureWarning` is disabled by setting
    ``cupy.disable_experimental_warning`` to ``True``,
    which is ``False`` by default.

    The basic usage is to call it in the function or method we want to
    mark as experimental along with the API name.

    .. testsetup::

        import sys
        import warnings

        warnings.simplefilter('always')

        def wrapper(message, category, filename, lineno, file=None, line=None):
            sys.stdout.write(warnings.formatwarning(
                message, category, filename, lineno))

        showwarning_orig = warnings.showwarning
        warnings.showwarning = wrapper

    .. testcleanup::

        warnings.showwarning = showwarning_orig

    .. testcode::

        from cupy import _util

        def f(x):
            _util.experimental('cupy.foo.bar.f')
            # concrete implementation of f follows

        f(1)

    .. testoutput::
        :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

        ... FutureWarning: cupy.foo.bar.f is experimental. \
The interface can change in the future. ...

    We can also make a whole class experimental. In that case,
    we should call this function in its ``__init__`` method.

    .. testcode::

        class C():
            def __init__(self):
              _util.experimental('cupy.foo.C')

        C()

    .. testoutput::
        :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

        ... FutureWarning: cupy.foo.C is experimental. \
The interface can change in the future. ...

    If we want to mark ``__init__`` method only, rather than class itself,
    it is recommended that we explicitly feed its API name.

    .. testcode::

        class D():
            def __init__(self):
                _util.experimental('D.__init__')

        D()

    .. testoutput::
        :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

        ...  FutureWarning: D.__init__ is experimental. \
The interface can change in the future. ...

    Currently, we do not have any sophisticated way to mark some usage of
    non-experimental function as experimental.
    But we can support such usage by explicitly branching it.

    .. testcode::

        def g(x, experimental_arg=None):
            if experimental_arg is not None:
                _util.experimental('experimental_arg of cupy.foo.g')

    Args:
        api_name(str): The name of an API marked as experimental.
    """

    if not cupy.disable_experimental_feature_warning:
        warnings.warn('{} is experimental. '
                      'The interface can change in the future.'.format(
                          api_name),
                      FutureWarning)


class PerformanceWarning(RuntimeWarning):
    """Warning that indicates possible performance issues."""


def check_array(obj, *, arg_name):
    """Checks if the given object is an array.

    This function raises :class:`TypeError` if ``obj`` is not an instance
    of :type:`cupy.ndarray`\\ .
    """
    if not isinstance(obj, cupy.ndarray):
        raise TypeError(
            '\'{}\' must be a cupy.ndarray object, not {}.'.format(
                arg_name, type(obj)))


# This code is to signal when the interpreter is in shutdown mode
# to prevent using globals that could be already deleted in
# objects `__del__` method
#
# This solution is taken from the Numba/llvmlite code
_shutting_down = [False]


@atexit.register
def _at_shutdown():
    _shutting_down[0] = True


def is_shutting_down(_shutting_down=_shutting_down):
    """
    Whether the interpreter is currently shutting down.
    For use in finalizers, __del__ methods, and similar; it is advised
    to early bind this function rather than look it up when calling it,
    since at shutdown module globals may be cleared.
    """
    return _shutting_down[0]


try:
    # On older NumPy versions, registering ml_dtypes loops can trigger
    # segfaults even when the inputs never use those dtypes.
    if numpy.lib.NumpyVersion(numpy.__version__) < "2.1.2":
        raise ImportError("NumPy had a critical result-type issue")
    import ml_dtypes
except ImportError:
    ml_dtypes = None


cdef _get_ml_dtype(str name):
    """The scalar type of an `ml_dtypes` dtype or None if unavailable.

    `complex32` and `bcomplex32` were only added in ml_dtypes 0.6, so an
    older version may provide `bfloat16` only.
    """
    if ml_dtypes is None:
        return None
    scalar_type = getattr(ml_dtypes, name, None)
    if scalar_type is None:
        return None
    return numpy.dtype(scalar_type).type


BF16 = _get_ml_dtype("bfloat16")
# Older ml_dtypes versions have unsafe cast behavior between the narrow
# complex types, which can select the wrong ufunc loop and return garbage.
C32 = BC32 = None
if (ml_dtypes is not None
        and numpy.lib.NumpyVersion(ml_dtypes.__version__) >= "0.7.0"):
    C32 = _get_ml_dtype("complex32")
    BC32 = _get_ml_dtype("bcomplex32")

# DType classes are distinct from the scalar classes above.
BF16_DTYPE = type(numpy.dtype(BF16)) if BF16 is not None else None
C32_DTYPE = type(numpy.dtype(C32)) if C32 is not None else None
BC32_DTYPE = type(numpy.dtype(BC32)) if BC32 is not None else None


def _make_loop_helper(scalar_type, str name, str dtype_name):
    """Build the ufunc loop helper for one of the `ml_dtypes` types.

    The returned helper generates type signatures to be used with unpacking:
    ``*bf16_loop(...)``.  ``in_types``/``out_types`` are either a count of
    arguments of ``dtype_name`` or precise type specifiers, which we convert
    to dtypes for now (to sit in for the `type(dtype)` which is in theory
    more spot-on but in practice the same).  For mixed signatures, `None` is
    an alias for ``dtype_name``.

    When the dtype is unavailable the helper returns an empty tuple, so that
    the loop is simply not registered.

    NOTE: Always insert a loop after the matching NumPy one (e.g. bfloat16
    after float16) to make sure the NumPy dtype is preferred for odd cases
    (such as int8 being -- probably incorrectly -- promoted to float16).
    """
    @cython.binding(True)
    def loop(in_types=1, out_types=1, code=None):
        if scalar_type is None:
            return ()

        if isinstance(in_types, int):
            in_tuple = (scalar_type,) * in_types
        else:
            in_tuple = tuple(
                scalar_type if t is None else t for t in in_types)

        if isinstance(out_types, int):
            out_tuple = (scalar_type,) * out_types
        else:
            out_tuple = tuple(
                scalar_type if t is None else t for t in out_types)

        signature = [in_tuple, out_tuple]
        return ((signature, code),)

    loop.__name__ = name
    loop.__qualname__ = name
    loop.__doc__ = f"Define the {dtype_name} loop for a ufunc."
    return loop


bf16_loop = _make_loop_helper(BF16, "bf16_loop", "bfloat16")

# Register complex32 before bcomplex32 (as float16 comes before bfloat16).
c32_loop = _make_loop_helper(C32, "c32_loop", "complex32")
bc32_loop = _make_loop_helper(BC32, "bc32_loop", "bcomplex32")
