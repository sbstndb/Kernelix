# Kernelix Examples

add_executable(example_simple_gemm simple_gemm.cpp)
target_link_libraries(example_simple_gemm PRIVATE Kernelix::kernelix)

add_executable(example_fused_linear fused_linear.cpp)
target_link_libraries(example_fused_linear PRIVATE Kernelix::kernelix)

add_executable(example_parallel parallel.cpp)
target_link_libraries(example_parallel PRIVATE Kernelix::kernelix)
