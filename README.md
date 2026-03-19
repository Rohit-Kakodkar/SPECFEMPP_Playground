# SFPP Playground

A playground for experimenting with SPECFEM++ computational kernels. This project explores various
parallelization strategies for gradient operators using Kokkos, including CPU and GPU
implementations.

## Features

- Multiple gradient operator implementations:
  - Serial (reference implementation)
  - Kokkos RangePolicy
  - Kokkos MDRangePolicy
  - Kokkos TeamPolicy with various scratch memory strategies
  - CUDA/CuTe optimized kernels
- Benchmarking suite for performance comparison
- Unit tests with GoogleTest

## Prerequisites

- CMake 3.21 or higher
- C++20 compatible compiler
- (Optional) CUDA toolkit for GPU support

## Installation

### Clone the repository

```bash
git clone https://github.com/Rohit-Kakodkar/SPECFEMPP_Playground.git
cd SPECFEMPP_Playground
```

### Build using CMake Presets

The project provides several CMake presets for different build configurations:

**CPU Release build (native architecture):**

```bash
cmake --preset release
cmake --build build/release
```

**CPU Debug build:**

```bash
cmake --preset debug
cmake --build build/debug
```

**NVIDIA GPU build (Ampere architecture):**

```bash
cmake --preset release-ampere80
cmake --build build/release-ampere80
```

### Manual CMake configuration

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DKokkos_ARCH_NATIVE=ON
make -j
```

## Running Tests

```bash
cd build/release
ctest --output-on-failure
```

## Running Benchmarks

```bash
./build/release/bin/gradient
```

## Project Structure

```
├── core/                    # Core library
│   ├── kernels/            # Compute kernels
│   │   └── gradient/       # Gradient operator implementations
│   ├── jacobian_matrix/    # Jacobian matrix utilities
│   ├── quadrature/         # Quadrature rules
│   └── wavefield/          # Wavefield data structures
├── benchmark/              # Benchmark suite
├── tests/                  # Unit tests
└── cmake/                  # CMake modules
```

## License

See LICENSE file for details.
