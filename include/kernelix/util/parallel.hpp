#pragma once

#include "../core/config.hpp"
#include <functional>
#include <algorithm>

#if defined(KERNELIX_HAS_TBB)
    #include <tbb/parallel_for.h>
    #include <tbb/parallel_reduce.h>
    #include <tbb/blocked_range.h>
    #include <tbb/blocked_range2d.h>
    #include <tbb/partitioner.h>
#endif

namespace kernelix::parallel {

#if defined(KERNELIX_HAS_TBB)

/// 1D parallel for loop using TBB
template<typename Func>
void parallel_for(std::size_t begin, std::size_t end, Func&& func) {
    if (!should_parallelize(end - begin)) {
        // Sequential fallback for small sizes
        for (std::size_t i = begin; i < end; ++i) {
            func(i);
        }
        return;
    }

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(begin, end),
        [&func](const tbb::blocked_range<std::size_t>& range) {
            for (std::size_t i = range.begin(); i < range.end(); ++i) {
                func(i);
            }
        }
    );
}

/// 1D parallel for with grain size control
template<typename Func>
void parallel_for(std::size_t begin, std::size_t end, std::size_t grain_size, Func&& func) {
    if (!should_parallelize(end - begin)) {
        for (std::size_t i = begin; i < end; ++i) {
            func(i);
        }
        return;
    }

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(begin, end, grain_size),
        [&func](const tbb::blocked_range<std::size_t>& range) {
            for (std::size_t i = range.begin(); i < range.end(); ++i) {
                func(i);
            }
        }
    );
}

/// 2D parallel for loop using TBB
template<typename Func>
void parallel_for_2d(
    std::size_t row_begin, std::size_t row_end,
    std::size_t col_begin, std::size_t col_end,
    Func&& func)
{
    const std::size_t total = (row_end - row_begin) * (col_end - col_begin);

    if (!should_parallelize(total)) {
        for (std::size_t i = row_begin; i < row_end; ++i) {
            for (std::size_t j = col_begin; j < col_end; ++j) {
                func(i, j);
            }
        }
        return;
    }

    tbb::parallel_for(
        tbb::blocked_range2d<std::size_t>(row_begin, row_end, col_begin, col_end),
        [&func](const tbb::blocked_range2d<std::size_t>& range) {
            for (std::size_t i = range.rows().begin(); i < range.rows().end(); ++i) {
                for (std::size_t j = range.cols().begin(); j < range.cols().end(); ++j) {
                    func(i, j);
                }
            }
        }
    );
}

/// 2D parallel for with grain size control
template<typename Func>
void parallel_for_2d(
    std::size_t row_begin, std::size_t row_end, std::size_t row_grain,
    std::size_t col_begin, std::size_t col_end, std::size_t col_grain,
    Func&& func)
{
    const std::size_t total = (row_end - row_begin) * (col_end - col_begin);

    if (!should_parallelize(total)) {
        for (std::size_t i = row_begin; i < row_end; ++i) {
            for (std::size_t j = col_begin; j < col_end; ++j) {
                func(i, j);
            }
        }
        return;
    }

    tbb::parallel_for(
        tbb::blocked_range2d<std::size_t>(
            row_begin, row_end, row_grain,
            col_begin, col_end, col_grain
        ),
        [&func](const tbb::blocked_range2d<std::size_t>& range) {
            for (std::size_t i = range.rows().begin(); i < range.rows().end(); ++i) {
                for (std::size_t j = range.cols().begin(); j < range.cols().end(); ++j) {
                    func(i, j);
                }
            }
        }
    );
}

/// Parallel reduction using TBB
template<typename T, typename Func, typename Reduce>
T parallel_reduce(std::size_t begin, std::size_t end, T identity, Func&& func, Reduce&& reduce) {
    if (!should_parallelize(end - begin)) {
        T result = identity;
        for (std::size_t i = begin; i < end; ++i) {
            result = reduce(result, func(i));
        }
        return result;
    }

    return tbb::parallel_reduce(
        tbb::blocked_range<std::size_t>(begin, end),
        identity,
        [&func, &reduce](const tbb::blocked_range<std::size_t>& range, T init) {
            T result = init;
            for (std::size_t i = range.begin(); i < range.end(); ++i) {
                result = reduce(result, func(i));
            }
            return result;
        },
        reduce
    );
}

/// Parallel for with blocked range (for manual tile processing)
template<typename Func>
void parallel_for_blocked(std::size_t begin, std::size_t end, std::size_t block_size, Func&& func) {
    if (!should_parallelize(end - begin)) {
        for (std::size_t block_start = begin; block_start < end; block_start += block_size) {
            std::size_t block_end = std::min(block_start + block_size, end);
            func(block_start, block_end);
        }
        return;
    }

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(begin, end, block_size),
        [&func](const tbb::blocked_range<std::size_t>& range) {
            func(range.begin(), range.end());
        }
    );
}

#else // Sequential fallback when TBB is not available

/// 1D sequential for loop (fallback)
template<typename Func>
void parallel_for(std::size_t begin, std::size_t end, Func&& func) {
    for (std::size_t i = begin; i < end; ++i) {
        func(i);
    }
}

/// 1D sequential for with grain size (ignored in sequential mode)
template<typename Func>
void parallel_for(std::size_t begin, std::size_t end, std::size_t /*grain_size*/, Func&& func) {
    for (std::size_t i = begin; i < end; ++i) {
        func(i);
    }
}

/// 2D sequential for loop (fallback)
template<typename Func>
void parallel_for_2d(
    std::size_t row_begin, std::size_t row_end,
    std::size_t col_begin, std::size_t col_end,
    Func&& func)
{
    for (std::size_t i = row_begin; i < row_end; ++i) {
        for (std::size_t j = col_begin; j < col_end; ++j) {
            func(i, j);
        }
    }
}

/// 2D sequential for with grain size (ignored in sequential mode)
template<typename Func>
void parallel_for_2d(
    std::size_t row_begin, std::size_t row_end, std::size_t /*row_grain*/,
    std::size_t col_begin, std::size_t col_end, std::size_t /*col_grain*/,
    Func&& func)
{
    for (std::size_t i = row_begin; i < row_end; ++i) {
        for (std::size_t j = col_begin; j < col_end; ++j) {
            func(i, j);
        }
    }
}

/// Sequential reduction (fallback)
template<typename T, typename Func, typename Reduce>
T parallel_reduce(std::size_t begin, std::size_t end, T identity, Func&& func, Reduce&& reduce) {
    T result = identity;
    for (std::size_t i = begin; i < end; ++i) {
        result = reduce(result, func(i));
    }
    return result;
}

/// Sequential for with blocked range (fallback)
template<typename Func>
void parallel_for_blocked(std::size_t begin, std::size_t end, std::size_t block_size, Func&& func) {
    for (std::size_t block_start = begin; block_start < end; block_start += block_size) {
        std::size_t block_end = std::min(block_start + block_size, end);
        func(block_start, block_end);
    }
}

#endif // KERNELIX_HAS_TBB

/// Helper for tiled GEMM parallelization
template<typename Func>
void parallel_gemm_tiles(
    std::size_t M, std::size_t N, std::size_t K,
    std::size_t tile_m, std::size_t tile_n, std::size_t tile_k,
    Func&& func)
{
    const std::size_t num_m_tiles = (M + tile_m - 1) / tile_m;
    const std::size_t num_n_tiles = (N + tile_n - 1) / tile_n;
    const std::size_t num_k_tiles = (K + tile_k - 1) / tile_k;

    // Parallelize over M and N tiles (outer loops)
    parallel_for_2d(
        0, num_m_tiles, 0, num_n_tiles,
        [&](std::size_t mi, std::size_t ni) {
            const std::size_t m_start = mi * tile_m;
            const std::size_t m_end = std::min(m_start + tile_m, M);
            const std::size_t n_start = ni * tile_n;
            const std::size_t n_end = std::min(n_start + tile_n, N);

            // K tiles are processed sequentially (accumulation)
            for (std::size_t ki = 0; ki < num_k_tiles; ++ki) {
                const std::size_t k_start = ki * tile_k;
                const std::size_t k_end = std::min(k_start + tile_k, K);
                func(m_start, m_end, n_start, n_end, k_start, k_end);
            }
        }
    );
}

} // namespace kernelix::parallel
