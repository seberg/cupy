from __future__ import annotations

import numpy
import pytest

import cupy
from cupy.cuda import runtime
from cupy import testing


if not runtime.is_hip and cupy.cuda.get_local_runtime_version() < 12020:
    # As for bfloat16, we may be able to enable this, but skip for now.
    pytest.skip(allow_module_level=True,
                reason="narrow complex types are missing some features")

if numpy.lib.NumpyVersion(numpy.__version__) < "2.1.2":
    pytest.skip(allow_module_level=True,
                reason="ml_dtypes not enabled due to NumPy bug.")

ml_dtypes = pytest.importorskip('ml_dtypes', minversion='0.7.0')

C32 = numpy.dtype(ml_dtypes.complex32)
BC32 = numpy.dtype(ml_dtypes.bcomplex32)

# The real dtype the components are stored in.  It is also the result dtype
# of the functions returning a real value (`absolute` and `angle`).
COMPONENT = {
    C32: numpy.dtype(numpy.float16),
    BC32: numpy.dtype(ml_dtypes.bfloat16),
}

# Tolerances must be fixed when the test is defined while the dtype is a
# parameter, so use the coarser of the two everywhere.
TOL = float(ml_dtypes.finfo(BC32).eps)

parametrize_dtype = pytest.mark.parametrize(
    'dtype', [C32, BC32], ids=['complex32', 'bcomplex32'])

# Interesting real/imaginary parts, combined into a grid of test values.
_PARTS = [0.0, 1.0, -1.0, 2.5, -0.5, 100.0,
          numpy.inf, -numpy.inf, numpy.nan]
_FINITE_PARTS = [p for p in _PARTS if numpy.isfinite(p)]
# Small values only, used where an overflowing result is likely to differ.
_SMALL_PARTS = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0]


def _values(dtype, parts=_PARTS, nonzero=False):
    """The grid of test values as a 1-D array of ``dtype``."""
    values = numpy.array(
        [complex(re, im) for re in parts for im in parts],
        dtype=numpy.complex64)
    values = values.astype(dtype)
    if nonzero:
        values = values[values != 0]
    return values


def _real_values(dtype, parts=_FINITE_PARTS):
    """The real part of the test grid, cast to ``dtype``."""
    values = numpy.array(
        [complex(re, im) for re in parts for im in parts]).real
    return values.astype(dtype)


def _widen(res, dtype):
    """Check the result dtype and widen the result for comparison.

    ``numpy.testing`` cannot deal with the ml_dtypes complex types (it fails
    to promote them with the Python float tolerances), so all tests compare
    in ``complex64``.  Both sides are rounded to ``dtype`` already, which
    makes widening exact.
    """
    assert res.dtype == dtype
    return res.astype(numpy.complex64)


@parametrize_dtype
@testing.numpy_cupy_allclose(rtol=0, atol=0)
def test_asarray(xp, dtype):
    return _widen(xp.asarray(_values(dtype)), dtype)


@parametrize_dtype
@testing.numpy_cupy_allclose(rtol=0, atol=0)
def test_copy_noncontiguous(xp, dtype):
    a = xp.asarray(_values(dtype)).reshape(len(_PARTS), len(_PARTS))
    return _widen(a[::2, ::-2].copy(), dtype)


@parametrize_dtype
@pytest.mark.parametrize('func', ['positive', 'negative', 'conjugate'])
@testing.numpy_cupy_allclose(rtol=0, atol=0)
def test_unary_exact(xp, dtype, func):
    a = xp.asarray(_values(dtype))
    return _widen(getattr(xp, func)(a), dtype)


@parametrize_dtype
@pytest.mark.parametrize('method', ['conj', 'conjugate'])
@testing.numpy_cupy_allclose(rtol=0, atol=0)
def test_conjugate_method(xp, dtype, method):
    a = xp.asarray(_values(dtype))
    result = getattr(a, method)()
    assert result is not a
    return _widen(result, dtype)


def _component_array(xp, dtype, layout):
    if layout == 'scalar':
        return xp.asarray(_values(dtype)[13])
    if layout == 'empty':
        return xp.empty((0, 3), dtype=dtype)
    a = xp.asarray(_values(dtype)).reshape(len(_PARTS), len(_PARTS))
    if layout == 'strided':
        return a[::2, ::-2]
    if layout == 'empty_view':
        return a[:0]
    return a


@parametrize_dtype
@pytest.mark.parametrize('part', ['real', 'imag'])
@pytest.mark.parametrize('layout', [
    'contiguous', 'strided', 'scalar', 'empty', 'empty_view'])
@testing.numpy_cupy_allclose(rtol=0, atol=0)
def test_component_getter(xp, dtype, part, layout):
    a = _component_array(xp, dtype, layout)
    result = getattr(a, part)
    assert result.dtype == COMPONENT[dtype]
    assert result.shape == a.shape
    assert result.strides == a.strides
    return result.astype(numpy.float32)


@parametrize_dtype
@pytest.mark.parametrize('part', ['real', 'imag'])
@pytest.mark.parametrize('layout', [
    'contiguous', 'strided', 'scalar', 'empty', 'empty_view'])
@pytest.mark.parametrize('through_view', [False, True])
@testing.numpy_cupy_allclose(rtol=0, atol=0)
def test_component_setter(xp, dtype, part, layout, through_view):
    a = _component_array(xp, dtype, layout)
    values = (xp.arange(a.shape[-1]).astype(COMPONENT[dtype])
              if a.ndim else xp.asarray(-2.0, dtype=COMPONENT[dtype]))
    if through_view:
        getattr(a, part)[...] = values
    else:
        setattr(a, part, values)
    return _widen(a, dtype)


@parametrize_dtype
@pytest.mark.parametrize('func', [
    'iscomplex', 'isreal', 'isfinite', 'isinf', 'isnan'])
@testing.numpy_cupy_array_equal()
def test_predicates(xp, dtype, func):
    a = xp.asarray(_values(dtype))
    if xp is numpy:
        # NumPy also relies on dtype.kind for iscomplex/isreal, but
        # bcomplex32 reports kind "V". Use the complex values as the oracle.
        a = a.astype(numpy.complex64)
    return getattr(xp, func)(a)


@parametrize_dtype
def test_complex_type_predicates(dtype):
    a = cupy.zeros((), dtype=dtype)
    assert cupy.iscomplexobj(a)
    assert not cupy.isrealobj(a)


@parametrize_dtype
def test_isclose_complex_components(dtype):
    values = numpy.array([1 + 2j, 1], dtype=numpy.complex64).astype(dtype)
    a = cupy.asarray(values)
    b = cupy.ones(2, dtype=numpy.float32)
    testing.assert_array_equal(cupy.isclose(a, b), [False, True])
    testing.assert_array_equal(cupy.isclose(b, a), [False, True])


@parametrize_dtype
@pytest.mark.parametrize('operation', ['ufunc_out', 'inplace', 'copyto'])
def test_reject_complex_to_real_same_kind(dtype, operation):
    for xp in (numpy, cupy):
        a = xp.asarray(numpy.array([1+2j]).astype(dtype))
        real = xp.ones(1, dtype=ml_dtypes.bfloat16)
        with pytest.raises(TypeError):
            if operation == 'ufunc_out':
                xp.add(a, a, out=real)
            elif operation == 'inplace':
                real += a
            else:
                xp.copyto(real, a, casting='same_kind')


@parametrize_dtype
@pytest.mark.parametrize('layout', [
    'contiguous', 'strided', 'scalar', 'empty', 'empty_view'])
@pytest.mark.parametrize('inplace', [False, True])
@testing.numpy_cupy_array_equal()
def test_byteswap(xp, dtype, layout, inplace):
    a = _component_array(xp, dtype, layout)
    if xp is numpy:
        # Swap the two real components independently, preserving NaN bits.
        return a.reshape(-1).view(numpy.uint16).byteswap()
    result = a.byteswap(inplace=inplace)
    if inplace:
        assert result.data.ptr == a.data.ptr
    return result.reshape(-1).view(numpy.uint16)


@parametrize_dtype
@pytest.mark.parametrize('method', ['square', 'reciprocal'])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_square_reciprocal(xp, dtype, method):
    a = xp.asarray(_values(dtype, _SMALL_PARTS, nonzero=True))
    return _widen(getattr(xp, method)(a), dtype)


@parametrize_dtype
@pytest.mark.parametrize('exponent', [2, 2.0, -1.0, 0.5])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_power_scalar_fast_paths(xp, dtype, exponent):
    a = xp.asarray(_values(dtype, _SMALL_PARTS, nonzero=True))
    return (a ** exponent).astype(numpy.complex64)


@parametrize_dtype
@pytest.mark.parametrize('func, values', [
    ('sum', [1 + 1j, (1 + 1j) * 2**-12, -1 - 1j]),
    ('prod', [(65 + 18j) / 64] * 3),
])
@pytest.mark.parametrize('explicit_dtype', [False, True])
def test_sumprod_accumulates_in_complex64(dtype, func, values, explicit_dtype):
    values = numpy.array(values, dtype=numpy.complex64).astype(dtype)
    expected = getattr(numpy, func)(values.astype(numpy.complex64))
    expected = expected.astype(dtype).astype(numpy.complex64)
    result = getattr(cupy, func)(cupy.asarray(values),
                                 dtype=dtype if explicit_dtype else None)
    testing.assert_array_equal(_widen(result, dtype), expected)


@parametrize_dtype
@pytest.mark.parametrize('func, expected', [
    ('nansum', 3 + 2j), ('nanprod', 1 + 7j),
])
@pytest.mark.parametrize('explicit_dtype', [False, True])
def test_nan_reductions_replace_components(dtype, func, expected,
                                           explicit_dtype):
    values = [complex(numpy.nan, 2), complex(3, numpy.nan)]
    a = cupy.asarray(numpy.array(values, dtype=numpy.complex64).astype(dtype))
    result = getattr(cupy, func)(a, dtype=dtype if explicit_dtype else None)
    testing.assert_array_equal(_widen(result, dtype), expected)


@parametrize_dtype
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_vdot(xp, dtype):
    a = xp.asarray(_values(dtype, _SMALL_PARTS))
    if xp is numpy:
        a = a.astype(numpy.complex64)
    return xp.vdot(a, a[::-1]).astype(numpy.complex64)


@parametrize_dtype
@pytest.mark.parametrize('axis', [None, 0, 1])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_norm(xp, dtype, axis):
    a = xp.asarray(_values(dtype, _SMALL_PARTS)).reshape(6, 6)
    if xp is numpy:
        a = a.astype(numpy.complex64)
    return xp.linalg.norm(a, axis=axis)


@parametrize_dtype
@pytest.mark.parametrize('func', ['inv', 'det', 'eigvals', 'solve'])
def test_linalg_unsupported(dtype, func):
    a = cupy.asarray(numpy.array([[1+2j, 0], [0, 2-1j]]).astype(dtype))
    with pytest.raises(TypeError, match='unsupported in linalg'):
        if func == 'solve':
            cupy.linalg.solve(a, cupy.ones(2))
        else:
            getattr(cupy.linalg, func)(a)


@parametrize_dtype
@pytest.mark.parametrize('mode', ['full', 'same', 'valid'])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_convolve(xp, dtype, mode):
    a = xp.asarray(_values(dtype, _SMALL_PARTS))
    if xp is numpy:
        a = a.astype(numpy.complex64)
    return xp.convolve(a, a[:3], mode=mode).astype(numpy.complex64)


@parametrize_dtype
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_interp(xp, dtype):
    a = xp.asarray(numpy.array([1+2j, -1+.5j, 3-1j]).astype(dtype))
    if xp is numpy:
        a = a.astype(numpy.complex64)
    return xp.interp(xp.asarray([-.5, .5, 1.5, 3.]), xp.arange(3), a)


@parametrize_dtype
@testing.numpy_cupy_allclose(rtol=0, atol=0)
def test_interp_widens_before_subtraction(xp, dtype):
    value = 60000 if dtype == C32 else 2e38
    z = complex(value, value)
    samples = numpy.array([-z, z]).astype(dtype)
    if xp is numpy:
        samples = samples.astype(numpy.complex128)
    return xp.interp(xp.asarray([0.5]), xp.arange(2), xp.asarray(samples))


@parametrize_dtype
@pytest.mark.parametrize('part', ['real', 'imag'])
def test_shares_memory_with_components(dtype, part):
    a = cupy.ones(4, dtype=dtype)
    view = getattr(a, part)
    other = a.imag if part == 'real' else a.real
    assert cupy.shares_memory(a, view)
    assert not cupy.shares_memory(view, other)


@parametrize_dtype
@pytest.mark.parametrize('imaginary', [0.0, 1.0])
@testing.numpy_cupy_allclose(rtol=0, atol=0)
def test_real_if_close(xp, dtype, imaginary):
    a = xp.asarray(numpy.array([1+imaginary*1j]).astype(dtype))
    if xp is numpy:
        return xp.real_if_close(a.astype(numpy.complex64), tol=0.01).astype(
            numpy.complex64)
    result = xp.real_if_close(a, tol=0.01)
    assert result.dtype == (COMPONENT[dtype] if imaginary == 0 else dtype)
    return result.astype(numpy.complex64)


@parametrize_dtype
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_sqrt(xp, dtype):
    # Both sides use the FreeBSD `csqrtf` so even infinities should match.
    a = xp.asarray(_values(dtype))
    return _widen(xp.sqrt(a), dtype)


@parametrize_dtype
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_absolute(xp, dtype):
    a = xp.asarray(_values(dtype))
    res = xp.absolute(a)
    assert res.dtype == COMPONENT[dtype]
    return res


@parametrize_dtype
def test_sign_uses_float32_magnitude(dtype):
    # CuPy currently computes the magnitude and division in float32.
    value = 60000 if dtype == C32 else 2e38
    values = numpy.array([0, complex(value, value), -1 + 6j],
                         dtype=numpy.complex64).astype(dtype)
    expected = numpy.sign(values.astype(numpy.complex64)).astype(dtype)
    result = cupy.sign(cupy.asarray(values))
    testing.assert_array_equal(_widen(result, dtype),
                               expected.astype(numpy.complex64))


@parametrize_dtype
@pytest.mark.parametrize('deg', [False, True])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_angle(xp, dtype, deg):
    a = xp.asarray(_values(dtype))
    if xp is numpy:
        # NumPy has no `arctan2` loop for the narrow complex types.
        res = numpy.angle(a.astype(numpy.complex64), deg=deg)
        res = res.astype(COMPONENT[dtype])
    else:
        res = xp.angle(a, deg=deg)
    assert res.dtype == COMPONENT[dtype]
    return res


@parametrize_dtype
@pytest.mark.parametrize('func', ['add', 'subtract'])
@testing.numpy_cupy_allclose(rtol=0, atol=0)
def test_binary_exact(xp, dtype, func):
    x = xp.asarray(_values(dtype))
    y = x[:, None]  # Broadcasting to test all combinations
    with numpy.errstate(all='ignore'):
        return _widen(getattr(xp, func)(x, y), dtype)


@parametrize_dtype
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_multiply(xp, dtype):
    # NOTE(seberg): NumPy applies the C99 rules for multiplying infinities
    # while CUDA uses the naive formula, so only test finite input.
    x = xp.asarray(_values(dtype, _FINITE_PARTS))
    y = x[:, None]
    with numpy.errstate(all='ignore'):
        return _widen(xp.multiply(x, y), dtype)


@parametrize_dtype
@pytest.mark.parametrize('func', ['divide', 'true_divide'])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_divide(xp, dtype, func):
    # NOTE(seberg): Division by zero (and of infinities) is another place
    # where the CUDA and NumPy algorithms disagree.
    x = xp.asarray(_values(dtype, _FINITE_PARTS))
    y = xp.asarray(_values(dtype, _FINITE_PARTS, nonzero=True))[:, None]
    with numpy.errstate(all='ignore'):
        return _widen(getattr(xp, func)(x, y), dtype)


@parametrize_dtype
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_power(xp, dtype):
    # Powers of zero and large results are computed differently (CUDA uses
    # `exp(y * log(x))`), so keep both base and result small and finite.
    x = xp.asarray(_values(dtype, _SMALL_PARTS, nonzero=True))
    y = xp.asarray(_values(dtype, _SMALL_PARTS))[:, None]
    with numpy.errstate(all='ignore'):
        return _widen(xp.power(x, y), dtype)


@parametrize_dtype
@pytest.mark.parametrize('func', ['add', 'multiply'])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_narrow_scalar(xp, dtype, func):
    a = xp.asarray(_values(dtype, _FINITE_PARTS))
    return _widen(getattr(xp, func)(a, dtype.type(-0.5 + 2j)), dtype)


@parametrize_dtype
@pytest.mark.parametrize('scalar', [2, 2.0, -0.5 + 2j])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_python_scalar(xp, dtype, scalar):
    # Weak scalars are resolved via `result_type()` in both libraries, so
    # CuPy tracks whatever ml_dtypes does here.  As of 0.6.0 a Python int
    # keeps the narrow dtype while a float or complex promotes to complex64.
    a = xp.asarray(_values(dtype, _FINITE_PARTS))
    res = a * scalar
    # Whatever NumPy picks for the same operation, CuPy has to agree.
    assert res.dtype == (numpy.zeros(1, dtype) * scalar).dtype
    return res.astype(numpy.complex64)


@parametrize_dtype
@pytest.mark.parametrize('to_dtype', [
    numpy.complex64, numpy.complex128, numpy.float16, numpy.float32,
    numpy.float64, ml_dtypes.bfloat16, numpy.int16, numpy.int32, numpy.bool_,
])
@pytest.mark.filterwarnings('ignore::cupy.exceptions.ComplexWarning')
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_astype_from_narrow(xp, dtype, to_dtype):
    # Non-finite values are undefined when casting to an integer.
    a = xp.asarray(_values(dtype, _FINITE_PARTS))
    return a.astype(to_dtype)


@parametrize_dtype
@pytest.mark.parametrize('from_dtype', [
    numpy.complex64, numpy.complex128, numpy.float16, numpy.float32,
    numpy.float64, ml_dtypes.bfloat16, numpy.int16, numpy.int32, numpy.bool_,
])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_astype_to_narrow(xp, dtype, from_dtype):
    if numpy.dtype(from_dtype).kind == 'c':
        values = _values(from_dtype, _FINITE_PARTS)
    else:
        values = _real_values(from_dtype)
    return _widen(xp.asarray(values).astype(dtype), dtype)


@parametrize_dtype
@testing.numpy_cupy_allclose(rtol=0, atol=0)
def test_astype_between_narrow(xp, dtype):
    to_dtype = BC32 if dtype == C32 else C32
    a = xp.asarray(_values(dtype))
    return _widen(a.astype(to_dtype), to_dtype)


@parametrize_dtype
@pytest.mark.parametrize('size', [0, 1])
def test_astype_complex_warning(dtype, size):
    a = cupy.ones(size, dtype=dtype)
    with pytest.warns(cupy.exceptions.ComplexWarning):
        a.astype(ml_dtypes.bfloat16)


@parametrize_dtype
def test_dlpack_roundtrip(dtype):
    a = cupy.asarray(_values(dtype))
    b = cupy.from_dlpack(a)
    assert b.dtype == dtype
    assert b.data.ptr == a.data.ptr
    testing.assert_array_equal(b.view(numpy.uint32), a.view(numpy.uint32))


def test_dlpack_bcomplex32_requires_compatible_version():
    a = cupy.zeros(1, dtype=BC32)
    with pytest.raises(BufferError, match='DLPack 1.4'):
        a.__dlpack__(max_version=(1, 3))


@parametrize_dtype
@pytest.mark.parametrize('func', [
    lambda x: x + x,
    lambda x: cupy.sqrt(x),
    lambda x: cupy.conjugate(x) * x,
    lambda x: x.astype(numpy.complex64),
])
@pytest.mark.parametrize('old_fusion', [True, False])
def test_fusion_basic(dtype, func, old_fusion):
    # NOTE(seberg): As of writing v14.0 no attempt made for old-fusion.
    arr = cupy.asarray(_values(dtype, _FINITE_PARTS))
    expected = func(arr)
    count = 0

    def count_calls(x):
        nonlocal count
        count += 1
        return func(x)

    if old_fusion:
        fused = cupy._core.fusion.Fusion(count_calls, 'c32_fuse_test')
    else:
        fused = cupy._core.new_fusion.Fusion(count_calls, 'c32_fuse_test')

    actual = fused(arr)
    # If old-fusion doesn't work, it would try again with new fusion and
    # count_calls would be called twice, so ensure that isn't the case.
    assert count == 1
    cupy.testing.assert_allclose(
        actual.astype(numpy.complex64), expected.astype(numpy.complex64),
        rtol=TOL, atol=TOL)


@parametrize_dtype
@pytest.mark.parametrize('old_fusion', [True, False])
def test_fusion_narrow_scalar(dtype, old_fusion):
    # A narrow complex scalar goes through `_get_cuda_scalar_repr()`.
    scalar = dtype.type(-0.5 + 2j)
    arr = cupy.asarray(_values(dtype, _FINITE_PARTS))
    expected = arr * scalar

    if old_fusion:
        fusion = cupy._core.fusion
    else:
        fusion = cupy._core.new_fusion
    fused = fusion.Fusion(lambda x: x * scalar, 'c32_fuse_scalar_test')

    actual = fused(arr)
    assert actual.dtype == dtype
    cupy.testing.assert_allclose(
        actual.astype(numpy.complex64), expected.astype(numpy.complex64),
        rtol=TOL, atol=TOL)
