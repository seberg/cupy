#ifndef CUPY_NUMPYSTRING_H
#define CUPY_NUMPYSTRING_H


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

    template<typename OT, int Olen>
    __host__ __device__ NumPyString& operator=(const double &other)
    {
        /* create temporary char string, since we may have a unicode one */
        const NumPyString<char, this->maxlen> charstr;
        const char *end = charstr.data + this->maxlen;

        std::to_chars_result res;
        res = std::to_chars(charstr.data, end, other);
        /* zero fill unused chunk */
        for (char *ptr = res.ptr; ptr < end; ptr++) {
            ptr[i] = 0;
        }

        *this = charstr;
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
