#pragma once

#include "cupy/carray.cuh"
#include "cupy/complex.cuh"

// Shared implementation of the narrow complex types (`complex32` and
// `bcomplex32`, see complex32.cuh and bcomplex32.cuh).  Neither has hardware
// support, so -- as for `bfloat16` -- everything is computed in single
// precision and only the final result is rounded back down.

namespace cupy {

template <typename T>
class narrow_complex {
private:
  T real_;
  T imag_;

public:
  typedef T value_type;

  // Note that this zeroes: `narrow_complex<T>()` is used as the zero value
  // when casting to bool (see `fix_cast_expr` in _kernel.pyx).
  __device__ narrow_complex() : real_(0.0f), imag_(0.0f) {}
  __device__ narrow_complex(float re) : real_(re), imag_(0.0f) {}
  __device__ narrow_complex(float re, float im) : real_(re), imag_(im) {}
  __device__ narrow_complex(const thrust::complex<float> &z)
      : real_(z.real()), imag_(z.imag()) {}

  // Explicit constructors from other numeric types
  template <typename U>
  explicit __device__ narrow_complex(const narrow_complex<U> &z)
      : real_(z.real()), imag_(z.imag()) {}

  explicit __device__ narrow_complex(bool v) : real_(float(v)), imag_(0.0f) {}
  explicit __device__ narrow_complex(double v)
      : real_(float(v)), imag_(0.0f) {}
  explicit __device__ narrow_complex(int v) : real_(float(v)), imag_(0.0f) {}
  explicit __device__ narrow_complex(unsigned int v)
      : real_(float(v)), imag_(0.0f) {}
  explicit __device__ narrow_complex(long long v)
      : real_(float(v)), imag_(0.0f) {}
  explicit __device__ narrow_complex(unsigned long long v)
      : real_(float(v)), imag_(0.0f) {}
  explicit __device__ narrow_complex(const thrust::complex<double> &z)
      : real_(float(z.real())), imag_(float(z.imag())) {}

  __device__ operator thrust::complex<float>() const {
    return thrust::complex<float>(float(real_), float(imag_));
  }

  // `thrust::complex<double>` can only be reached via a second user defined
  // conversion, which C++ does not do, so provide it directly.
  __device__ operator thrust::complex<double>() const {
    return thrust::complex<double>(float(real_), float(imag_));
  }

  // Spelled out version of the above for use where an implicit conversion
  // would be ambiguous or is not considered at all.
  __device__ thrust::complex<float> as_complex() const {
    return thrust::complex<float>(float(real_), float(imag_));
  }

  __device__ float real() const { return float(real_); }
  __device__ float imag() const { return float(imag_); }

  // Generally allow assignments if explicit conversion is available
  // We may want a better solution for this (e.g. fix kernels to explicitly
  // convert)
  narrow_complex& operator=(const narrow_complex&) = default;

  template <typename U>
  __device__ narrow_complex& operator=(const U &rhs) {
    *this = static_cast<narrow_complex>(rhs);
    return *this;
  }

  template <typename U>
  __device__ narrow_complex& operator+=(const U &rhs) {
    *this = *this + rhs;
    return *this;
  }

  template <typename U>
  __device__ narrow_complex& operator-=(const U &rhs) {
    *this = *this - rhs;
    return *this;
  }

  template <typename U>
  __device__ narrow_complex& operator*=(const U &rhs) {
    *this = *this * rhs;
    return *this;
  }

  template <typename U>
  __device__ narrow_complex& operator/=(const U &rhs) {
    *this = *this / rhs;
    return *this;
  }

  // Special value checking methods
  __device__ int iszero() const { return real_.iszero() && imag_.iszero(); }
  __device__ int isnan() const { return real_.isnan() || imag_.isnan(); }
  __device__ int isinf() const { return real_.isinf() || imag_.isinf(); }
  __device__ int isfinite() const {
    return real_.isfinite() && imag_.isfinite();
  }
};


template <typename T>
__device__ narrow_complex<T> operator+(const narrow_complex<T> &z) {
  return z;
}

template <typename T>
__device__ narrow_complex<T> operator-(const narrow_complex<T> &z) {
  return narrow_complex<T>(-z.real(), -z.imag());
}

// The `thrust::complex` operators are templates, so they are never found for
// a `narrow_complex` operand (no implicit conversion happens during template
// argument deduction).  Spell out the overloads we need instead.
#define CUPY_NARROW_COMPLEX_BINARY_OP(OP)                                    \
  template <typename T>                                                      \
  __device__ narrow_complex<T> operator OP(                                  \
      const narrow_complex<T> &a, const narrow_complex<T> &b) {              \
    return narrow_complex<T>(a.as_complex() OP b.as_complex());              \
  }                                                                          \
  template <typename T>                                                      \
  __device__ narrow_complex<T> operator OP(                                  \
      const narrow_complex<T> &a, const thrust::complex<float> &b) {         \
    return narrow_complex<T>(a.as_complex() OP b);                           \
  }                                                                          \
  template <typename T>                                                      \
  __device__ narrow_complex<T> operator OP(                                  \
      const thrust::complex<float> &a, const narrow_complex<T> &b) {         \
    return narrow_complex<T>(a OP b.as_complex());                           \
  }                                                                          \
  template <typename T>                                                      \
  __device__ narrow_complex<T> operator OP(                                  \
      const narrow_complex<T> &a, float b) {                                 \
    return narrow_complex<T>(a.as_complex() OP thrust::complex<float>(b));   \
  }                                                                          \
  template <typename T>                                                      \
  __device__ narrow_complex<T> operator OP(                                  \
      float a, const narrow_complex<T> &b) {                                 \
    return narrow_complex<T>(thrust::complex<float>(a) OP b.as_complex());   \
  }

CUPY_NARROW_COMPLEX_BINARY_OP(+)
CUPY_NARROW_COMPLEX_BINARY_OP(-)
CUPY_NARROW_COMPLEX_BINARY_OP(*)
CUPY_NARROW_COMPLEX_BINARY_OP(/)

#undef CUPY_NARROW_COMPLEX_BINARY_OP

#define CUPY_NARROW_COMPLEX_EQUALITY_OP(OP, NEGATE)                          \
  template <typename T>                                                      \
  __device__ bool operator OP(                                               \
      const narrow_complex<T> &a, const narrow_complex<T> &b) {              \
    return NEGATE(a.real() == b.real() && a.imag() == b.imag());             \
  }                                                                          \
  template <typename T>                                                      \
  __device__ bool operator OP(                                               \
      const narrow_complex<T> &a, const thrust::complex<float> &b) {         \
    return NEGATE(a.real() == b.real() && a.imag() == b.imag());             \
  }                                                                          \
  template <typename T>                                                      \
  __device__ bool operator OP(                                               \
      const thrust::complex<float> &a, const narrow_complex<T> &b) {         \
    return NEGATE(a.real() == b.real() && a.imag() == b.imag());             \
  }                                                                          \
  template <typename T>                                                      \
  __device__ bool operator OP(const narrow_complex<T> &a, float b) {         \
    return NEGATE(a.real() == b && a.imag() == 0.0f);                        \
  }                                                                          \
  template <typename T>                                                      \
  __device__ bool operator OP(float a, const narrow_complex<T> &b) {         \
    return NEGATE(a == b.real() && b.imag() == 0.0f);                        \
  }

CUPY_NARROW_COMPLEX_EQUALITY_OP(==, )
CUPY_NARROW_COMPLEX_EQUALITY_OP(!=, !)

#undef CUPY_NARROW_COMPLEX_EQUALITY_OP

template <typename T>
__device__ float real(const narrow_complex<T> &z) {
  return z.real();
}

template <typename T>
__device__ float imag(const narrow_complex<T> &z) {
  return z.imag();
}

template <typename T>
__device__ float abs(const narrow_complex<T> &z) {
  return thrust::abs(z.as_complex());
}

template <typename T>
__device__ float arg(const narrow_complex<T> &z) {
  return thrust::arg(z.as_complex());
}

template <typename T>
__device__ float norm(const narrow_complex<T> &z) {
  return thrust::norm(z.as_complex());
}

template <typename T>
__device__ narrow_complex<T> conj(const narrow_complex<T> &z) {
  return narrow_complex<T>(z.real(), -z.imag());
}

// Elementwise functions, all computed in single precision.
#define CUPY_NARROW_COMPLEX_UNARY_FUNC(NAME)                                 \
  template <typename T>                                                      \
  __device__ narrow_complex<T> NAME(const narrow_complex<T> &z) {            \
    return narrow_complex<T>(thrust::NAME(z.as_complex()));                  \
  }

CUPY_NARROW_COMPLEX_UNARY_FUNC(sqrt)
CUPY_NARROW_COMPLEX_UNARY_FUNC(exp)
CUPY_NARROW_COMPLEX_UNARY_FUNC(log)
CUPY_NARROW_COMPLEX_UNARY_FUNC(log10)
CUPY_NARROW_COMPLEX_UNARY_FUNC(sin)
CUPY_NARROW_COMPLEX_UNARY_FUNC(cos)
CUPY_NARROW_COMPLEX_UNARY_FUNC(tan)
CUPY_NARROW_COMPLEX_UNARY_FUNC(sinh)
CUPY_NARROW_COMPLEX_UNARY_FUNC(cosh)
CUPY_NARROW_COMPLEX_UNARY_FUNC(tanh)
CUPY_NARROW_COMPLEX_UNARY_FUNC(asin)
CUPY_NARROW_COMPLEX_UNARY_FUNC(acos)
CUPY_NARROW_COMPLEX_UNARY_FUNC(atan)
CUPY_NARROW_COMPLEX_UNARY_FUNC(asinh)
CUPY_NARROW_COMPLEX_UNARY_FUNC(acosh)
CUPY_NARROW_COMPLEX_UNARY_FUNC(atanh)

#undef CUPY_NARROW_COMPLEX_UNARY_FUNC

// log1p, log2, expm1 and rint are provided by cupy/complex.cuh at global scope
#define CUPY_NARROW_COMPLEX_UNARY_CUPY_FUNC(NAME)                            \
  template <typename T>                                                      \
  __device__ narrow_complex<T> NAME(const narrow_complex<T> &z) {            \
    return narrow_complex<T>(::NAME(z.as_complex()));                        \
  }

CUPY_NARROW_COMPLEX_UNARY_CUPY_FUNC(log1p)
CUPY_NARROW_COMPLEX_UNARY_CUPY_FUNC(log2)
CUPY_NARROW_COMPLEX_UNARY_CUPY_FUNC(expm1)
CUPY_NARROW_COMPLEX_UNARY_CUPY_FUNC(rint)

#undef CUPY_NARROW_COMPLEX_UNARY_CUPY_FUNC

template <typename T>
__device__ narrow_complex<T> pow(
    const narrow_complex<T> &x, const narrow_complex<T> &y) {
  return narrow_complex<T>(thrust::pow(x.as_complex(), y.as_complex()));
}

template <typename T>
__device__ narrow_complex<T> pow(const narrow_complex<T> &x, float y) {
  return narrow_complex<T>(thrust::pow(x.as_complex(), y));
}

template <typename T>
__device__ narrow_complex<T> pow(float x, const narrow_complex<T> &y) {
  return narrow_complex<T>(thrust::pow(x, y.as_complex()));
}

// Lexicographic min/max with NumPy's NaN handling, see cupy/complex.cuh
template <typename T>
__device__ narrow_complex<T> min(
    const narrow_complex<T> &x, const narrow_complex<T> &y) {
  return narrow_complex<T>(::min(x.as_complex(), y.as_complex()));
}

template <typename T>
__device__ narrow_complex<T> max(
    const narrow_complex<T> &x, const narrow_complex<T> &y) {
  return narrow_complex<T>(::max(x.as_complex(), y.as_complex()));
}

// Special value checking functions (free functions for compatibility)
template <typename T>
__device__ int iszero(const narrow_complex<T> &z) {
  return z.iszero();
}

template <typename T>
__device__ int isnan(const narrow_complex<T> &z) {
  return z.isnan();
}

template <typename T>
__device__ int isinf(const narrow_complex<T> &z) {
  return z.isinf();
}

template <typename T>
__device__ int isfinite(const narrow_complex<T> &z) {
  return z.isfinite();
}

}  // namespace cupy
