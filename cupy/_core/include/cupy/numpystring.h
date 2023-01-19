#ifndef CUPY_NUMPYSTRING_H
#define CUPY_NUMPYSTRING_H


template<typename T, int maxlen>
class NumPyString {
public:
    T data[maxlen];

    __host__ __device__ int strlen() {
        int len = maxlen;
        while (len > 0 && this->data[len-1] == 0) {
            len--;
        }
        return len;
    }

    __host__ __device__ T operator[](int i){
        /* Allowing too large `i` for easier handling of different length */
        if (i < this.maxlen) {
            return this.data[i];
        }
        return 0;
    }

    __host__ __device__ NumPyString& operator=(NumPyString &other)
    {
        // For now don't allow, we should error if not ascii/latin1, or
        // should it simply cast and don't worry about it?
        static_assert(sizeof(this.T) >= sizeof(other.T));
        for (int i = 0; i < this.maxlen; i++) {
            this.data[i] = other[i];
        }
        return this;
    }

    __host__ __device__ bool operator==(NumPyString &other)
    {
        int longer = this.maxlen > other.maxlen ? this.maxlen : other.maxlen;
        for (int i = 0; i < longer; i++) {
            if (this->data[i] != other.data[i]) {
                return false;
            }
        }
        return true;
    }
};


#endif  // CUPY_NUMPYSTRING_H