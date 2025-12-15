# distutils: language = c++


cdef class _MyFuture:
    cdef object _event
    cdef object _exception
    cdef object _result

    cpdef set_exception(self, e)
    cpdef set_result(self, result)
    cpdef wait_result(self)


cdef class ExactlyOnceDict(dict):
    cdef inline get(self, key, default=None):
        result = dict.get(self, key, default)
        if type(result) is _MyFuture:
            return result.wait_result()
        return result

    cpdef setdefault_once(self, key, default_func)
