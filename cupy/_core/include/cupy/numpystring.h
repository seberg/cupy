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
        // TODO: This is very unsafe (as it just casts)
        for (int i = 0; i < this->maxlen; i++) {
            this->data[i] = other[i];
        }
        return *this;
    }

    template<typename OT, int Olen>
    __host__ __device__ bool operator==(const NumPyString<OT, Olen> &other)
    {
        int longer = this->maxlen > other->maxlen ? this->maxlen : other->maxlen;
        for (int i = 0; i < longer; i++) {
            if (this->data[i] != other->data[i]) {
                return false;
            }
        }
        return true;
    }
};


#endif  // CUPY_NUMPYSTRING_H