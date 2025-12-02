# Kernelix Tests

# Try to find Google Test
find_package(GTest QUIET)

if(GTest_FOUND)
    message(STATUS "Found GTest, building tests with GTest")

    # Test executable
    add_executable(kernelix_tests
        test_tensor.cpp
        test_expr.cpp
        test_gemm.cpp
        test_fusion.cpp
        test_parallel.cpp
    )

    target_link_libraries(kernelix_tests PRIVATE
        Kernelix::kernelix
        GTest::gtest
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(kernelix_tests)

else()
    message(STATUS "GTest not found, building minimal test runner")

    # Minimal test without GTest
    add_executable(kernelix_tests
        test_minimal.cpp
    )

    target_link_libraries(kernelix_tests PRIVATE
        Kernelix::kernelix
    )

    add_test(NAME kernelix_minimal_tests COMMAND kernelix_tests)
endif()
