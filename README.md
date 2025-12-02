# Kernelix

**Kernelix** is a C++ header-only library for automatic tensor kernel optimization targeting CPU inference, primarily designed for LLM workloads.

## Philosophy

The user describes *what* to compute via an expressive API, and Kernelix determines *how* to execute it optimally — without external code generation, without JIT, everything resolved at compile-time via expression templates and metaprogramming.

```cpp
// User writes naturally:
auto output = kernelix::linear(kernelix::rmsnorm(x), W, bias);
kernelix::eval(output, y_ptr);

// Kernelix automatically generates an optimized fused kernel
```

## Core Principles

- **Zero-overhead abstraction**: Expressivity costs nothing at runtime
- **Automatic fusion**: Patterns are detected and fused without intervention
- **Portability**: Standard C++ with optional intrinsics, no heavy dependencies
- **Extensibility**: Adding an operator is natural and automatically benefits from the optimization system
- **Modern parallelism**: TBB-based parallel execution for multi-core scalability

## Requirements

- **C++20** compatible compiler (GCC 11+, Clang 13+, MSVC 2022+)
- **CMake** 3.16 or higher
- **Intel TBB** (Threading Building Blocks) for parallelism

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `KERNELIX_BUILD_TESTS` | `ON` | Build unit tests |
| `KERNELIX_BUILD_BENCHMARKS` | `OFF` | Build benchmarks |
| `KERNELIX_BUILD_EXAMPLES` | `ON` | Build examples |
| `KERNELIX_ENABLE_AVX2` | `ON` | Enable AVX2 intrinsics |
| `KERNELIX_ENABLE_AVX512` | `OFF` | Enable AVX-512 intrinsics |
| `KERNELIX_ENABLE_NEON` | Auto | Enable ARM NEON intrinsics |

### Running Tests

```bash
cd build
ctest --output-on-failure
```

## Quick Start

```cpp
#include <kernelix/kernelix.hpp>

int main() {
    using namespace kernelix;

    // Static dimensions for maximum performance
    Tensor<float, 32, 4096> x;
    Tensor<float, 4096, 4096> W;
    Tensor<float, 4096> bias;

    // Build expression (lazy evaluation)
    auto result = linear(rmsnorm(x), W, bias);

    // Evaluate - Kernelix optimizes automatically:
    // - Fused rmsnorm + linear kernel
    // - Optimal tiling for cache hierarchy
    // - Parallel execution via TBB
    Tensor<float, 32, 4096> output;
    eval(result, output.data());

    return 0;
}
```

## Architecture

```
+----------------------------------------------------------+
|                    USER LAYER                            |
|  Expressive API: linear(), rmsnorm(), softmax(), etc.    |
+----------------------------------------------------------+
                          |
                          v
+----------------------------------------------------------+
|                EXPRESSION TEMPLATES                       |
|  Each operation returns a type (not a value)             |
|  Expression tree encoded in the type system              |
+----------------------------------------------------------+
                          |
                          v
+----------------------------------------------------------+
|                 ANALYSIS & TRAITS                        |
|  ExprTraits<E>: extracts op_type, shapes, flags          |
|  PatternMatcher<E>: detects fuseable patterns            |
+----------------------------------------------------------+
                          |
                          v
+----------------------------------------------------------+
|                    SCHEDULING                            |
|  TileSelector: optimal tile sizes                        |
|  FusionPolicy: fusion decisions                          |
|  TBB parallelization strategy                            |
+----------------------------------------------------------+
                          |
                          v
+----------------------------------------------------------+
|                  MICRO-KERNELS                           |
|  Vectorized implementations (AVX2/AVX-512/NEON)          |
|  TBB parallel_for for multi-threaded execution           |
+----------------------------------------------------------+
```

## Supported Operations

### Contractions
- `gemm(A, B)` - General matrix multiplication
- `gemv(A, x)` - Matrix-vector multiplication
- `dot(a, b)` - Dot product
- `batch_gemm(A, B)` - Batched GEMM

### Linear Layers
- `linear(x, W)` - Linear transformation
- `linear(x, W, bias)` - Linear with bias

### Normalizations
- `rmsnorm(x, weight, eps)` - RMS normalization
- `layernorm(x, gamma, beta)` - Layer normalization

### Activations
- `relu(x)` - ReLU activation
- `silu(x)` - SiLU (Swish) activation
- `gelu(x)` - GELU activation

### Attention
- `scaled_dot_attention(Q, K, V)` - Scaled dot-product attention
- `softmax(x, axis)` - Softmax

### Positional Encoding
- `rope(x, cos, sin)` - Rotary Position Embedding

## Automatic Fusion Patterns

Kernelix automatically detects and fuses common patterns:

| Pattern | Description |
|---------|-------------|
| `GemmBiasAct` | GEMM + Bias + Activation |
| `RMSNormLinear` | RMSNorm followed by Linear |
| `SwiGLU` | SiLU(xW1) * xW2 |
| `FusedAttention` | Softmax(QK^T/sqrt(d)) @ V |
| `ResidualAdd` | x + sublayer(x) |

## Project Structure

```
kernelix/
├── CMakeLists.txt
├── README.md
├── include/
│   └── kernelix/
│       ├── kernelix.hpp          # Single include header
│       ├── core/                 # Tensors, concepts, config
│       ├── expr/                 # Expression templates
│       ├── traits/               # Type traits
│       ├── patterns/             # Pattern matching
│       ├── schedule/             # Scheduling & cost model
│       ├── kernel/               # Micro-kernels
│       ├── api/                  # User-facing API
│       └── util/                 # SIMD, parallel (TBB), memory
├── tests/
├── benchmarks/
└── examples/
```

## Extending Kernelix

Adding a new operator is straightforward:

```cpp
// 1. Define the expression
namespace kernelix::expr {
    template<typename Input>
    struct MyOpExpr : ExprBase {
        Input input;
        float param;
    };
}

// 2. Define traits
namespace kernelix::traits {
    template<typename I>
    struct ExprTraits<expr::MyOpExpr<I>> {
        using op_category = Elementwise;
        static constexpr auto output_shape = ExprTraits<I>::output_shape;
        // ...
    };
}

// 3. Implement evaluator (kernel)
namespace kernelix::kernel {
    template<typename Input>
    struct Evaluator<expr::MyOpExpr<Input>> {
        static void run(const expr::MyOpExpr<Input>& e, float* out) {
            // TBB-parallelized implementation
        }
    };
}

// 4. User API
namespace kernelix {
    auto my_op(auto&& e, float param) {
        return expr::MyOpExpr{std::forward<decltype(e)>(e), param};
    }
}
```

## Non-Goals

- Not a training framework (no autograd)
- Not a full polyhedral compiler
- Not a GPU runtime (CPU only)
- Not a dynamic graph manager

## License

MIT License

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.
