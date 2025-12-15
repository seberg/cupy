# distutils: language = c++

import atexit
import functools
import os
import threading
import warnings

import cython

import cupy
from cupy_backends.cuda.api cimport runtime


DEF CYTHON_BUILD_VER = CUPY_CYTHON_VERSION
cython_build_ver = CYTHON_BUILD_VER


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


cdef class _MyFuture:
    def __cinit__(self):
        self._event = threading.Event()

    cpdef set_exception(self, e):
        self._exception = e
        self._event.set()

    cpdef set_result(self, result):
        self._result = result
        self._event.set()

    cpdef wait_result(self):
        self._event.wait()
        if self._exception is not None:
            raise self._exception
        return self._result


cdef list _memos = []


cdef class ExactlyOnceDict(dict):
    """Dictionary that has a ``setdefault_once`` method that is similar to
    ``setdefault`` but ensures only one thread calls it's creation function.

    This is a crude implementation with the hope that future Python versions
    (3.15+) may provide a clearer pattern.
    An alternative would be to only allow a single thread to add new entries
    via an RW-lock.
    It further assumes that the function is either very rarely called (mostly
    cache hits) or expensive.

    Why? The original reason for this pattern is to prevent unloading of
    modules. Without the "exactly once", multiple threads may compile and
    load the same code. Except one thread, others will then immediately
    unload again when done.
    This is mostly fine but breaks graph capture where the loaded code is
    clearly still needed later.

    Deleting/clearing should be fine, but not well ordered.
    """
    def __getitem__(self, key):
        raise NotImplementedError("__getitem__(): use `.get()`")

    def __setitem__(self, key, value):
        raise NotImplementedError("__setitem__(): use `.setdefault_once()`")

    cpdef setdefault_once(self, key, default_func):
        """Similar to dict.setdefault but guarantees that the function is
        only evaluated if this thread get's to store the result in the dict.
        """
        cdef dict selfd = <dict>self
        my_future = _MyFuture()
        result = selfd.setdefault(key, my_future)
        if result is not my_future:
            if type(result) is _MyFuture:
                return result.wait_result()
            else:
                return result

        # Our event is set, we get to do the actual work and signal
        # those threads waiting on us (on error, waiting threads will fail).
        try:
            result = default_func()
            # Store result on event, so waiting threads do not have to trust
            # dict, which yet another thread could be deleting.
            selfd[key] = result
            my_future.set_result(result)
            return result
        except BaseException as e:
            # This part is only no-crash thread-safe. If someone cleared the
            # dict we could be deleting something that isn't our future.
            my_future.set_exception(e)
            selfd.pop(key, None)
            raise


def memoize(bint for_each_device=False):
    """Makes a function memoizing the result for each argument and device.

    This decorator provides automatic memoization of the function result.
    Unlike python's functools.cache, this decorator will evaluate the function
    exactly once per cache entry.
    This is helpful when creating kernels, see ``ExactlyOnceDict``.

    Args:
        for_each_device (bool): If ``True``, it memoizes the results for each
            device. Otherwise, it memoizes the results only based on the
            arguments.

    """
    def decorator(f):
        memo = ExactlyOnceDict()
        _memos.append(memo)

        @functools.wraps(f)
        @cython.binding(True)
        def ret(*args, **kwargs):
            cdef int id = -1
            cdef ExactlyOnceDict m = <ExactlyOnceDict>memo
            if for_each_device:
                id = runtime.getDevice()
            if len(kwargs):
                arg_key = (id, args, frozenset(kwargs.items()))
            else:
                arg_key = (id, args)
            result = m.get(arg_key, m)
            if result is m:
                result = m.setdefault_once(
                    arg_key, lambda: f(*args, **kwargs))

            return result

        return ret

    return decorator


@atexit.register
def clear_memo():
    """Clears the memoized results for all functions decorated by memoize.

    .. note::
        Clearing the cache may not be safe in all contexts. For example it
        may cause crashes when used in a program that uses graph capturing.
    """
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
