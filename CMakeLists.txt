cmake_minimum_required(VERSION 3.21)

# Project metadata
project(
  SFPP_Playground
  VERSION 0.1.0
  DESCRIPTION
    "Playground for experimenting with SPECFEM++ computational kernels"
  LANGUAGES CXX)

# ============================================================================
# CMake Options and Settings
# ============================================================================

# Only set default build type for single-config generators
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

# Options
option(BUILD_TESTING "Build tests" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_BENCHMARKS "Build benchmark suite" ON)
option(ENABLE_UNITY_BUILD "Enable unity (jumbo) builds for faster compilation"
       OFF)
option(BUILD_DOCS "Build documentation using Doxygen" OFF)
option(ENABLE_CLANG_TIDY "Enable clang-tidy checks during build" ON)

# Generate compile_commands.json for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Position Independent Code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Global Settings
# ============================================================================

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ============================================================================
# Dependencies
# ============================================================================

include(FetchContent)

# Set FetchContent defaults
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# Google Test (if testing enabled)
if(BUILD_TESTING)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE)
  # Prevent overriding the parent project's compiler/linker settings on Windows
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# Google Benchmark (if benchmarking enabled)
if(BUILD_BENCHMARKS)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
    GIT_SHALLOW TRUE)
  set(BENCHMARK_ENABLE_TESTING
      OFF
      CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL
      OFF
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/kokkos.cmake)

# ============================================================================
# Compiler Warnings
# ============================================================================

add_library(sfpp_playground_warnings INTERFACE)
add_library(SFPP_Playground::warnings ALIAS sfpp_playground_warnings)

if(MSVC)
  target_compile_options(
    sfpp_playground_warnings
    INTERFACE /W4 # High warning level
              /WX # Treat warnings as errors (optional, comment out if too
                  # strict)
              /permissive- # Standards conformance
  )
else()
  target_compile_options(
    sfpp_playground_warnings
    INTERFACE -Wall
              -Wextra
              -Wpedantic
              -Wshadow
              -Wnon-virtual-dtor
              -Wcast-align
              -Wunused
              -Woverloaded-virtual
              -Wnull-dereference
              -Wdouble-promotion
              -Wformat=2)

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(
      sfpp_playground_warnings
      INTERFACE -Wmisleading-indentation -Wduplicated-cond
                -Wduplicated-branches -Wlogical-op)
  endif()
endif()

# ============================================================================
# Clang-Tidy Integration
# ============================================================================

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
  else()
    message(WARNING "clang-tidy not found!")
  endif()
endif()

# ============================================================================
# Add Subdirectories
# ============================================================================

add_subdirectory(core)

# # Future subdirectories can be added here: if(BUILD_EXAMPLES)
# add_subdirectory(examples) endif()

# if(BUILD_TESTING) enable_testing() add_subdirectory(tests) endif()

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmark)
endif()

# ============================================================================
# Documentation
# ============================================================================

if(BUILD_DOCS)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    add_subdirectory(docs)
  else()
    message(WARNING "Doxygen not found, documentation will not be built")
  endif()
endif()

# ============================================================================
# Print Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "========== SFPP Playground Configuration ==========")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  BUILD_TESTING:       ${BUILD_TESTING}")
message(STATUS "  BUILD_EXAMPLES:      ${BUILD_EXAMPLES}")
message(STATUS "  BUILD_BENCHMARKS:    ${BUILD_BENCHMARKS}")
message(STATUS "  BUILD_DOCS:          ${BUILD_DOCS}")
message(STATUS "  ENABLE_UNITY_BUILD:  ${ENABLE_UNITY_BUILD}")
message(STATUS "  ENABLE_CLANG_TIDY:   ${ENABLE_CLANG_TIDY}")
message(STATUS "===================================================")
message(STATUS "")
