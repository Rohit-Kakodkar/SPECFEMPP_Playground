# ============================================================================
# Tests
# ============================================================================

# Find GTest
find_package(GTest REQUIRED)

# Create the gradient test executable
add_executable(gradient_test gradient.cpp)

# Link against required libraries
target_link_libraries(gradient_test 
    PRIVATE 
        SFPP_Playground::core
        SFPP_Playground::warnings
        Kokkos::kokkos
        GTest::gtest
        GTest::gtest_main
)

# Set output directory
set_target_properties(gradient_test PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

# Disable clang-tidy for test files
set_target_properties(gradient_test PROPERTIES CXX_CLANG_TIDY "")

# Add the test to CTest
include(GoogleTest)
gtest_discover_tests(gradient_test)