#ifndef CUPY_NUMPYSTRING_H
#define CUPY_NUMPYSTRING_H


/* The code below is heavily inspired by the cuDF version */
template<typename IntT, typename CharT>
__host__ __device__ void
integer_to_string(const IntT value, int strlen, CharT *ptr_orig)
{
    /* create temporary char string to flip later */
    char digits[cuda::std::numeric_limits<IntT>::digits10];

    bool const is_negative = value < 0;
    IntT absval = is_negative ? -value : value;

    int digits_idx = 0;
    do {
        digits[digits_idx++] = '0' + absval % (IntT)10;
        // next digit
        absval = absval / (IntT)10;
    } while (absval != 0);

    CharT *ptr = ptr_orig;
    if (is_negative) {
        *ptr++ = '-';
    }
    // digits are backwards, reverse the string into the output
    while (digits_idx-- > 0 && strlen-- > 0) {
        *ptr++ = digits[digits_idx];
    }

    /* zero fill unused chunk */
    for (; ptr < ptr_orig + strlen; ptr++) {
        *ptr = 0;
    }
}


template<typename T, int maxlen_>
class NumPyString {
public:
    static const int maxlen = maxlen_;
    // TODO: 0 is possible, but C doesn't like it here (for good reasons)
    //       there may be a way to tell C++ that empty is OK/possible?
    T data[maxlen_ ? maxlen_ : 1];

    __host__ __device__ int strlen() {
        int len = maxlen;
        while (len > 0 && this->data[len-1] == 0) {
            len--;
        }
        return len;
    }

    __host__ __device__ NumPyString () {}

    template<typename OT, int Olen>
    __host__ __device__ NumPyString (const NumPyString<OT, Olen> &other)
    {
        // TODO: This is very unsafe (as it just casts)
        for (int i = 0; i < this->maxlen; i++) {
            this->data[i] = other[i];
        }
    }

    __host__ __device__ T operator[](int i) const {
        /* Allowing too large `i` for easier handling of different length */
        if (i < this->maxlen) {
            return this->data[i];
        }
        return 0;
    }

    template<typename OT, int Olen>
    __host__ __device__ NumPyString& operator=(const NumPyString<OT, Olen> &other)
    {
        // NOTE: Unlike NumPy, we just cast U->S (unsafe).
        for (int i = 0; i < this->maxlen; i++) {
            this->data[i] = other[i];
        }
        return *this;
    }

    // TODO: Howto template it for all integers?!
    __host__ __device__ NumPyString& operator=(const long &value)
    {
        integer_to_string(value, maxlen_, this->data);
        return *this;
    }

    template<typename OT, int Olen>
    __host__ __device__ bool operator==(const NumPyString<OT, Olen> &other) const
    {
        int longer = this->maxlen > other.maxlen ? this->maxlen : other.maxlen;
        for (int i = 0; i < longer; i++) {
            if ((*this)[i] != other[i]) {
                return false;
            }
        }
        return true;
    }
};


#endif  // CUPY_NUMPYSTRING_H

