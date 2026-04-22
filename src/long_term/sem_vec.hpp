#pragma once

#include <cstddef>
#include <cstdlib>
#include <stdfloat>
#include <cstdint>

struct SemVec {
    size_t dimensions;
    std::float16_t* data;

    SemVec(size_t dimensions) : dimensions(dimensions) {
        data = static_cast<std::float16_t*>(std::malloc(dimensions * sizeof(std::float16_t)));
    }

    ~SemVec() {
        std::free(data);
    }
};
