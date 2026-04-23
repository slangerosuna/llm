#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
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

    SemVec(size_t dimensions) : dimensions(dimensions), data(nullptr) {
        data = static_cast<float16_t*>(std::malloc(dimensions * sizeof(float16_t)));
        if (data == nullptr) {
            dimensions = 0;
            return;
        }
    }

    SemVec(const SemVec& other) : dimensions(other.dimensions), data(nullptr) {
        data = static_cast<float16_t*>(std::malloc(dimensions * sizeof(float16_t)));
        if (data == nullptr) {
            dimensions = 0;
            return;
        }
        std::memcpy(data, other.data, dimensions * sizeof(float16_t));
    }

    SemVec(SemVec&& other) noexcept : dimensions(other.dimensions), data(other.data) {
        other.dimensions = 0;
        other.data = nullptr;
    }

    SemVec& operator=(const SemVec& other) {
        if (this == &other) {
            return *this;
        }

        float16_t* new_data = static_cast<float16_t*>(std::malloc(other.dimensions * sizeof(float16_t)));
        if (new_data != nullptr && other.data != nullptr) {
            std::memcpy(new_data, other.data, other.dimensions * sizeof(float16_t));
        }

        std::free(data);
        data = new_data;
        dimensions = (new_data != nullptr) ? other.dimensions : 0;
        return *this;
    }

    SemVec& operator=(SemVec&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        std::free(data);
        dimensions = other.dimensions;
        data = other.data;

        other.dimensions = 0;
        other.data = nullptr;
        return *this;
    }

    ~SemVec() {
        std::free(data);
    }
};
