#include <SFPP_playground.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace sfpp_playground;

template <typename ParallelizationStrategy, typename WavefieldInitializer,
          typename QuadratureInitializer, typename JacobianInitializer, typename Layout,
          typename ExecSpace>
class GradientDriver {
public:
    using FieldType = Wavefield<WavefieldInitializer, Layout, ExecSpace>;
    using QuadratureType = Quadrature<QuadratureInitializer, Layout, ExecSpace>;
    using JacobianType = JacobianMatrix2D<JacobianInitializer, Layout, ExecSpace>;
    using GradientType = Gradient<ParallelizationStrategy, FieldType, QuadratureType, JacobianType>;

    GradientDriver(const size_t n_elements, const size_t ngll, const size_t ncomponents)
        : field_(WavefieldInitializer{n_elements, ngll, ngll, ncomponents}),
          lprime_(QuadratureInitializer{ngll}), J_(JacobianInitializer{n_elements, ngll, ngll}),
          gradient_(ParallelizationStrategy{}, field_, lprime_, J_) {
    }

    void warmup() {
        size_t n_warmup = 100;
        for (size_t i = 0; i < n_warmup; ++i) {
            gradient_();
        }
    }

    void benchmark() {
        warmup();
        auto start = std::chrono::high_resolution_clock::now();
        auto grad = gradient_();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Strategy: " << std::setw(30) << std::left << GradientType::name()
                  << " | Time: " << std::fixed << std::setprecision(6) << elapsed.count()
                  << " seconds" << std::endl;
    }

private:
    FieldType field_;
    QuadratureType lprime_;
    JacobianType J_;
    GradientType gradient_;
};

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        using namespace sfpp_playground;

        constexpr size_t n_elements = 1 << 15;
        constexpr size_t ngll = 8;
        constexpr size_t ncomponents = 1;

        using layout_left = Kokkos::LayoutLeft;
        using layout_right = Kokkos::LayoutRight;

        // Range
        {
            GradientDriver<RangeTag, WavefieldZeroInitializer2D, QuadratureIdentityInitializer,
                           JacobianMatrixRegularInitializer2D, layout_left,
                           Kokkos::DefaultExecutionSpace>
                driver(n_elements, ngll, ncomponents);
            driver.benchmark();
        }

        // MDRange
        {
            GradientDriver<MDRangeTag, WavefieldZeroInitializer2D, QuadratureIdentityInitializer,
                           JacobianMatrixRegularInitializer2D, layout_left,
                           Kokkos::DefaultExecutionSpace>
                driver(n_elements, ngll, ncomponents);
            driver.benchmark();
        }

        // TeamPolicy
        {
            GradientDriver<TeamPolicyTag, WavefieldZeroInitializer2D, QuadratureIdentityInitializer,
                           JacobianMatrixRegularInitializer2D, layout_left,
                           Kokkos::DefaultExecutionSpace>
                driver(n_elements, ngll, ncomponents);
            driver.benchmark();
        }

        // TeamPolicy with Scratch View
        {
            GradientDriver<TeamPolicyWScratchVTag, WavefieldZeroInitializer2D,
                           QuadratureIdentityInitializer, JacobianMatrixRegularInitializer2D,
                           layout_left, Kokkos::DefaultExecutionSpace>
                driver(n_elements, ngll, ncomponents);
            driver.benchmark();
        }

        // TeamPolicy with Chunked Scratch View
        {
            GradientDriver<TeamPolicyWChunkedScratchVTag, WavefieldZeroInitializer2D,
                           QuadratureIdentityInitializer, JacobianMatrixRegularInitializer2D,
                           layout_left, Kokkos::DefaultExecutionSpace>
                driver(n_elements, ngll, ncomponents);
            driver.benchmark();
        }

        // TeamPolicy with Tiled Scratch View
        {
            GradientDriver<TeamPolicyWTiledScratchVTag, WavefieldZeroInitializer2D,
                           QuadratureIdentityInitializer, JacobianMatrixRegularInitializer2D,
                           layout_left, Kokkos::DefaultExecutionSpace>
                driver(n_elements, ngll, ncomponents);
            driver.benchmark();
        }

        // CUTE-based Implementation
        // {
        //     GradientDriver<CuteImplementationTag, WavefieldZeroInitializer2D,
        //                    QuadratureIdentityInitializer, JacobianMatrixRegularInitializer2D,
        //                    layout_left, Kokkos::DefaultExecutionSpace>
        //         driver(n_elements, ngll, ncomponents);
        //     driver.benchmark();
        // }

        // CUTE-based Implementation with Tiled Copy and FMA
        {
            GradientDriver<CuteCopyFMATag, WavefieldZeroInitializer2D,
                           QuadratureIdentityInitializer, JacobianMatrixRegularInitializer2D,
                           layout_left, Kokkos::DefaultExecutionSpace>
                driver(n_elements, ngll, ncomponents);
            driver.benchmark();
        }
    }
    Kokkos::finalize();
    return 0;
}
