#pragma once

#include <cstddef>
#include <cstdlib>
#include <stdfloat>
#include <cstdint>

#if defined(__STDCPP_FLOAT16_T__)
using float16_t = std::float16_t;
#elif defined(__FLT16_MANT_DIG__)
using float16_t = _Float16;
#else
#error "float16_t is not supported by this compiler"
#endif

struct SemVec {
    size_t dimensions;
    float16_t* data;

    SemVec(size_t dimensions) : dimensions(dimensions) {
        data = static_cast<float16_t*>(std::malloc(dimensions * sizeof(float16_t)));
    }

    ~SemVec() {
        std::free(data);
    }
};
