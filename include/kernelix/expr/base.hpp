#pragma once

#include "../core/types.hpp"

namespace kernelix::expr {

/// Base tag for all expressions (for concept detection)
struct ExprBase {};

/// Operation tags
struct AddOp {};
struct SubOp {};
struct MulOp {};
struct DivOp {};

/// Activation tags
struct ReluOp {};
struct SiluOp {};
struct GeluOp {};
struct TanhOp {};

/// Normalization tags
struct RMSNormTag {};
struct LayerNormTag {};

/// Fusion tags
struct NoFusion {};
struct FuseTag {};

} // namespace kernelix::expr
