#include <SFPP_playground.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace sfpp_playground;

template <typename GradientType>
void warmup(const GradientType& gradient) {
    constexpr size_t n_warmup_iterations = 10;
    for (size_t i = 0; i < n_warmup_iterations; ++i) {
        auto grad = gradient();
    }
}

template <typename GradientType>
void benchmark(const GradientType& gradient) {
    auto start = std::chrono::high_resolution_clock::now();
    auto grad = gradient();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Strategy: " << std::setw(30) << std::left << GradientType::name()
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

        const auto serial = Gradient(SerialTag{}, field, lprime, J);
        const auto md_range = Gradient(MDRangeTag{}, field, lprime, J);
        const auto team_policy = Gradient(TeamPolicyTag{}, field, lprime, J);
        const auto team_policy_scratch = Gradient(TeamPolicyWScratchVTag{}, field, lprime, J);
        const auto team_policy_chunked_scratch =
            Gradient(TeamPolicyWChunkedScratchVTag{}, field, lprime, J);

        // Compute gradient using different strategies
        warmup(serial);
        warmup(md_range);
        warmup(team_policy);
        warmup(team_policy_scratch);
        warmup(team_policy_chunked_scratch);
        // Benchmarking code can be added here
        benchmark(serial);
        benchmark(md_range);
        benchmark(team_policy);
        benchmark(team_policy_scratch);
        benchmark(team_policy_chunked_scratch);
    }
    Kokkos::finalize();
    return 0;
}
