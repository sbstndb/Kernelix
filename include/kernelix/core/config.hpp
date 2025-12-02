#pragma once

#include <cstddef>
#include <memory>
#include <thread>

#if defined(KERNELIX_HAS_TBB)
    #include <tbb/global_control.h>
    #include <tbb/info.h>
#endif

namespace kernelix {

/// Hardware configuration (detected or user-specified)
struct HardwareConfig {
    std::size_t l1_size = 32 * 1024;        // 32 KB
    std::size_t l2_size = 256 * 1024;       // 256 KB
    std::size_t l3_size = 8 * 1024 * 1024;  // 8 MB
    std::size_t cache_line = 64;
    std::size_t vector_width = 8;           // floats for AVX-256
    std::size_t num_threads = 0;            // 0 = auto (TBB default or hardware_concurrency)
};

/// Global configuration
struct Config {
    HardwareConfig hardware;
    bool enable_prefetch = true;
    std::size_t l2_tile_target = 0;         // 0 = auto
    bool enable_parallel = true;            // Enable parallelism (TBB if available)
    std::size_t parallel_threshold = 4096;  // Min elements for parallel execution
};

namespace detail {
    inline Config& global_config() {
        static Config cfg;
        return cfg;
    }

#if defined(KERNELIX_HAS_TBB)
    inline std::unique_ptr<tbb::global_control>& tbb_control() {
        static std::unique_ptr<tbb::global_control> gc;
        return gc;
    }
#endif
}

/// Set global configuration
inline void set_config(const Config& cfg) {
    detail::global_config() = cfg;

#if defined(KERNELIX_HAS_TBB)
    // Configure TBB thread count if specified
    if (cfg.hardware.num_threads > 0) {
        detail::tbb_control() = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism,
            cfg.hardware.num_threads
        );
    }
#endif
}

/// Get current configuration
inline const Config& get_config() {
    return detail::global_config();
}

/// Get number of available threads
inline std::size_t num_threads() {
    const auto& cfg = get_config();
    if (cfg.hardware.num_threads > 0) {
        return cfg.hardware.num_threads;
    }
#if defined(KERNELIX_HAS_TBB)
    return static_cast<std::size_t>(tbb::info::default_concurrency());
#else
    return static_cast<std::size_t>(std::thread::hardware_concurrency());
#endif
}

/// Check if parallel execution should be used for given size
inline bool should_parallelize(std::size_t size) {
    const auto& cfg = get_config();
    return cfg.enable_parallel && size >= cfg.parallel_threshold;
}

} // namespace kernelix
