#pragma once

#include "cupy/carray.cuh"
#include "cupy/float16.cuh"
#include "cupy/narrow_complex.cuh"

// A complex number made up of two `float16` values, matching
// `ml_dtypes.complex32`.  See narrow_complex.cuh for the implementation.
typedef cupy::narrow_complex<float16> complex32;
