#include <SFPP_playground.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace sfpp_playground;

template <typename ParallelizationStrategy, typename FieldType, typename QuadratureType,
          typename JacobianType>
void warmup(const ParallelizationStrategy /*unused*/, const FieldType& field,
            const QuadratureType& lprime, const JacobianType& J) {
    constexpr size_t n_warmup_iterations = 10;
    for (size_t i = 0; i < n_warmup_iterations; ++i) {
        auto gradient = Gradient(ParallelizationStrategy{}, field, lprime, J)();
    }
}

template <typename ParallelizationStrategy, typename FieldType, typename QuadratureType,
          typename JacobianType>
void benchmark(const ParallelizationStrategy /*unused*/, const FieldType& field,
               const QuadratureType& lprime, const JacobianType& J) {
    auto start = std::chrono::high_resolution_clock::now();
    auto gradient = Gradient(ParallelizationStrategy{}, field, lprime, J)();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Strategy: " << std::setw(30) << std::left << ParallelizationStrategy::name()
              << " | Time: " << std::fixed << std::setprecision(6) << elapsed.count() << " seconds"
              << std::endl;
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        using namespace sfpp_playground;

        constexpr int n_elements = 1 << 15;
        constexpr int ngll = 5;
        constexpr int ncomponents = 1;

        // Initialize wavefield
        Wavefield<WavefieldZeroInitializer2D> field{
            WavefieldZeroInitializer2D{n_elements, ngll, ngll, ncomponents}};

        // Initialize quadrature
        Quadrature<QuadratureIdentityInitializer> lprime{QuadratureIdentityInitializer{ngll}};

        // Initialize Jacobian matrix
        JacobianMatrix2D<JacobianMatrixRegularInitializer2D> J{
            JacobianMatrixRegularInitializer2D{n_elements, ngll, ngll}};

        // Compute gradient using different strategies
        warmup(SerialTag{}, field, lprime, J);
        warmup(MDRangeTag{}, field, lprime, J);
        warmup(TeamPolicyTag{}, field, lprime, J);
        warmup(TeamPolicyWScratchVTag{}, field, lprime, J);
        warmup(TeamPolicyWChunkedScratchVTag{}, field, lprime, J);

        // Benchmarking code can be added here
        benchmark(SerialTag{}, field, lprime, J);
        benchmark(MDRangeTag{}, field, lprime, J);
        benchmark(TeamPolicyTag{}, field, lprime, J);
        benchmark(TeamPolicyWScratchVTag{}, field, lprime, J);
        benchmark(TeamPolicyWChunkedScratchVTag{}, field, lprime, J);
    }
    Kokkos::finalize();
    return 0;
}
