#pragma once

#include "cupy/carray.cuh"
#include "cupy/bfloat16.cuh"
#include "cupy/narrow_complex.cuh"

// A complex number made up of two `bfloat16` values, matching
// `ml_dtypes.bcomplex32`.  See narrow_complex.cuh for the implementation.
typedef cupy::narrow_complex<bfloat16> bcomplex32;
