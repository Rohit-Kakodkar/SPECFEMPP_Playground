# ============================================================================
# Examples
# ============================================================================

add_executable(gradient gradient.cpp)

target_link_libraries(gradient PRIVATE SFPP_Playground::core
                                       SFPP_Playground::warnings
                                       Kokkos::kokkos)

# Disable clang-tidy for this target
set_target_properties(gradient PROPERTIES CXX_CLANG_TIDY "")

# Set output directory
set_target_properties(gradient PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                               ${CMAKE_BINARY_DIR}/bin)
# ============================================================================
