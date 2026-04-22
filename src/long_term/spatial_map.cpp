#include <long_term/spatial_map.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

typedef unsigned __int128 uint128_t;

static inline uint64_t rotl64(uint64_t x, int8_t r) {
    return (x << r) | (x >> (64 - r));
}

static uint128_t hash_bytes(const uint8_t* data, size_t len) {
    uint64_t h1 = 0;
    uint64_t h2 = 0;

    const uint64_t c1 = 0x87c37b91114253d5ULL;
    const uint64_t c2 = 0x4cf5ad432745937fULL;

    const size_t nblocks = len / 16;
    const uint64_t* blocks = reinterpret_cast<const uint64_t*>(data);

    for (size_t i = 0; i < nblocks; ++i) {
        uint64_t k1 = blocks[2 * i + 0];
        uint64_t k2 = blocks[2 * i + 1];

        k1 *= c1;
        k1 = rotl64(k1, 31);
        k1 *= c2;
        h1 ^= k1;
        h1 = rotl64(h1, 27);
        h1 += h2;
        h1 = h1 * 5 + 0x52dce729;

        k2 *= c2;
        k2 = rotl64(k2, 33);
        k2 *= c1;
        h2 ^= k2;
        h2 = rotl64(h2, 31);
        h2 += h1;
        h2 = h2 * 5 + 0x38495ab5;
    }

    const uint8_t* tail = data + nblocks * 16;
    uint64_t k1 = 0;
    uint64_t k2 = 0;

    switch (len & 15U) {
        case 15:
            k2 ^= static_cast<uint64_t>(tail[14]) << 48;
            [[fallthrough]];
        case 14:
            k2 ^= static_cast<uint64_t>(tail[13]) << 40;
            [[fallthrough]];
        case 13:
            k2 ^= static_cast<uint64_t>(tail[12]) << 32;
            [[fallthrough]];
        case 12:
            k2 ^= static_cast<uint64_t>(tail[11]) << 24;
            [[fallthrough]];
        case 11:
            k2 ^= static_cast<uint64_t>(tail[10]) << 16;
            [[fallthrough]];
        case 10:
            k2 ^= static_cast<uint64_t>(tail[9]) << 8;
            [[fallthrough]];
        case 9:
            k2 ^= static_cast<uint64_t>(tail[8]);
            k2 *= c2;
            k2 = rotl64(k2, 33);
            k2 *= c1;
            h2 ^= k2;
            [[fallthrough]];
        case 8:
            k1 ^= static_cast<uint64_t>(tail[7]) << 56;
            [[fallthrough]];
        case 7:
            k1 ^= static_cast<uint64_t>(tail[6]) << 48;
            [[fallthrough]];
        case 6:
            k1 ^= static_cast<uint64_t>(tail[5]) << 40;
            [[fallthrough]];
        case 5:
            k1 ^= static_cast<uint64_t>(tail[4]) << 32;
            [[fallthrough]];
        case 4:
            k1 ^= static_cast<uint64_t>(tail[3]) << 24;
            [[fallthrough]];
        case 3:
            k1 ^= static_cast<uint64_t>(tail[2]) << 16;
            [[fallthrough]];
        case 2:
            k1 ^= static_cast<uint64_t>(tail[1]) << 8;
            [[fallthrough]];
        case 1:
            k1 ^= static_cast<uint64_t>(tail[0]);
            k1 *= c1;
            k1 = rotl64(k1, 31);
            k1 *= c2;
            h1 ^= k1;
            [[fallthrough]];
        default:
            break;
    }

    h1 ^= len;
    h2 ^= len;

    h1 += h2;
    h2 += h1;

    auto fmix = [](uint64_t k) {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53ULL;
        k ^= k >> 33;
        return k;
    };

    h1 = fmix(h1);
    h2 = fmix(h2);

    h1 += h2;
    h2 += h1;

    return (static_cast<uint128_t>(h2) << 64) | h1;
}

static uint8_t* quantize(const SemVec& sem_vec, uint8_t splits_per_dimension) {
    const size_t n = sem_vec.dimensions;
    uint8_t* quantized = static_cast<uint8_t*>(std::malloc(n));
    if (quantized == nullptr) {
        return nullptr;
    }

    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 scale = _mm256_set1_ps(static_cast<float>(splits_per_dimension));
    const __m256 minv = _mm256_set1_ps(0.0f);
    const __m256 maxv = _mm256_set1_ps(static_cast<float>(splits_per_dimension));

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m128i half_values = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(&sem_vec.data[i]));
        __m256 v = _mm256_cvtph_ps(half_values);
        v = _mm256_add_ps(v, one);
        v = _mm256_mul_ps(v, half);
        v = _mm256_mul_ps(v, scale);
        v = _mm256_max_ps(v, minv);
        v = _mm256_min_ps(v, maxv);

        __m256i vi = _mm256_cvttps_epi32(v);

        __m128i lo = _mm256_castsi256_si128(vi);
        __m128i hi = _mm256_extracti128_si256(vi, 1);
        __m128i packed16 = _mm_packus_epi32(lo, hi);
        __m128i packed8 = _mm_packus_epi16(packed16, packed16);

        *reinterpret_cast<uint64_t*>(quantized + i) = static_cast<uint64_t>(_mm_cvtsi128_si64(packed8));
    }

    for (; i < n; ++i) {
        const float x = static_cast<float>(sem_vec.data[i]);
        const float norm = std::clamp((x + 1.0f) * 0.5f, 0.0f, 1.0f);
        quantized[i] = static_cast<uint8_t>(norm * static_cast<float>(splits_per_dimension));
    }

    return quantized;
}

static uint128_t hash_semvec(const SemVec& sem_vec, uint8_t splits_per_dimension) {
    const size_t n = sem_vec.dimensions;
    uint8_t* quantized = quantize(sem_vec, splits_per_dimension);
    if (quantized == nullptr) {
        return 0;
    }

    const uint128_t hash_value = hash_bytes(quantized, n);
    std::free(quantized);
    return hash_value;
}

struct NeighborTiles {
    const SemVec& sem_vec;
    const uint8_t splits_per_dimension;
    uint8_t* base_tile;
    uint8_t* working_tile;
    size_t dim_index;
    bool emit_plus;
    bool emitted_base;
    float cur_distance;

    NeighborTiles(const SemVec& sem_vec, uint8_t splits_per_dimension)
      : sem_vec(sem_vec),
        splits_per_dimension(splits_per_dimension),
        base_tile(quantize(sem_vec, splits_per_dimension)),
        working_tile(nullptr),
        dim_index(0),
        emit_plus(true),
        emitted_base(false),
        cur_distance(0.0f) {
        if (base_tile != nullptr) {
            working_tile = static_cast<uint8_t*>(std::malloc(sem_vec.dimensions));
            if (working_tile != nullptr) {
                std::memcpy(working_tile, base_tile, sem_vec.dimensions);
            }
        }
    }

    ~NeighborTiles() {
        std::free(base_tile);
        std::free(working_tile);
    }

    // tile position, distance to the tile after this
    std::pair<uint8_t*, float> next_tile() {
        if (base_tile == nullptr || working_tile == nullptr || sem_vec.dimensions == 0) {
            return {nullptr, std::numeric_limits<float>::infinity()};
        }

        if (!emitted_base) {
            emitted_base = true;
            cur_distance = 0.0f;
            return {base_tile, 1.0f};
        }

        while (dim_index < sem_vec.dimensions) {
            const uint8_t center = base_tile[dim_index];

            if (emit_plus) {
                emit_plus = false;
                if (center < splits_per_dimension) {
                    std::memcpy(working_tile, base_tile, sem_vec.dimensions);
                    working_tile[dim_index] = static_cast<uint8_t>(center + 1U);
                    cur_distance = 1.0f;
                    return {working_tile, 1.0f};
                }
            }

            emit_plus = true;
            ++dim_index;
            if (center > 0U) {
                std::memcpy(working_tile, base_tile, sem_vec.dimensions);
                working_tile[dim_index - 1] = static_cast<uint8_t>(center - 1U);
                cur_distance = 1.0f;
                return {working_tile, 1.0f};
            }
        }

        return {nullptr, std::numeric_limits<float>::infinity()};
    }
};

static inline float l2_distance_sq(const SemVec& lhs, const SemVec& rhs) {
    const size_t dims = std::min(lhs.dimensions, rhs.dimensions);
    float acc = 0.0f;

    for (size_t i = 0; i < dims; ++i) {
        const float lhs_value = static_cast<float>(lhs.data[i]);
        const float rhs_value = static_cast<float>(rhs.data[i]);
        const float d = lhs_value - rhs_value;
        acc += d * d;
    }

    return acc;
}

bool SpatialMap::should_grow(size_t target_bucket_size) const {
    const bool heavy_bucket = target_bucket_size > bucket_size;
    const bool high_global_load = (entry_count + 1) > (buckets.size() * bucket_size);
    return heavy_bucket || high_global_load;
}

void SpatialMap::rehash(size_t new_bucket_count) {
    new_bucket_count = std::max<size_t>(new_bucket_count, 1);
    std::vector<std::vector<std::pair<size_t, SemVec*>>> new_buckets(new_bucket_count);

    for (const auto& bucket : buckets) {
        for (const auto& [node_index, sem_vec] : bucket) {
            if (sem_vec == nullptr) {
                continue;
            }

            const uint128_t hv = hash_semvec(*sem_vec, splits_per_dimension);
            const size_t new_index = static_cast<size_t>(hv % new_bucket_count);
            new_buckets[new_index].emplace_back(node_index, sem_vec);
        }
    }

    buckets = std::move(new_buckets);
}

void SpatialMap::insert(size_t node_index, SemVec* sem_vec) {
    if (sem_vec == nullptr || buckets.empty()) {
        return;
    }

    const uint128_t hv = hash_semvec(*sem_vec, splits_per_dimension);
    size_t bucket_index = static_cast<size_t>(hv % buckets.size());

    if (should_grow(buckets[bucket_index].size() + 1)) {
        rehash(buckets.size() * 2);
        bucket_index = static_cast<size_t>(hv % buckets.size());
    }

    auto& bucket = buckets[bucket_index];
    bucket.emplace_back(node_index, sem_vec);
    ++entry_count;
}

void SpatialMap::get_nearest_neighbors(
    SemVec* query_vec,
    size_t k,
    std::vector<std::pair<size_t, SemVec*>>& neighbors) {
    neighbors.clear();
    if (query_vec == nullptr || k == 0 || buckets.empty()) {
        return;
    }

    using Candidate = std::pair<float, std::pair<size_t, SemVec*>>;
    std::priority_queue<Candidate> best;

    auto consider_bucket = [&](const std::vector<std::pair<size_t, SemVec*>>& bucket) {
        for (const auto& [node_index, sem_vec] : bucket) {
            if (sem_vec == nullptr) {
                continue;
            }

            const float distance = l2_distance_sq(*query_vec, *sem_vec);
            if (best.size() < k) {
                best.push({distance, {node_index, sem_vec}});
                continue;
            }

            if (distance < best.top().first) {
                best.pop();
                best.push({distance, {node_index, sem_vec}});
            }
        }
    };

    std::unordered_set<size_t> visited;
    visited.reserve(std::min(buckets.size(), query_vec->dimensions * 2 + 1));

    NeighborTiles tiles(*query_vec, splits_per_dimension);
    while (true) {
        auto [tile, _next_distance] = tiles.next_tile();
        (void)_next_distance;
        if (tile == nullptr) {
            break;
        }

        const uint128_t hv = hash_bytes(tile, query_vec->dimensions);
        const size_t bucket_index = static_cast<size_t>(hv % buckets.size());

        if (visited.insert(bucket_index).second) {
            consider_bucket(buckets[bucket_index]);
        }
    }

    if (best.size() < k) {
        for (size_t i = 0; i < buckets.size(); ++i) {
            if (visited.contains(i)) {
                continue;
            }
            consider_bucket(buckets[i]);
        }
    }

    neighbors.reserve(best.size());
    while (!best.empty()) {
        neighbors.push_back(best.top().second);
        best.pop();
    }
    std::reverse(neighbors.begin(), neighbors.end());
}
