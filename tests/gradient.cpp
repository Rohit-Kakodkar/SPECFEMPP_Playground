#include "KokkosEnvironment.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <SFPP_playground.hpp>
#include <cmath>

using namespace sfpp_playground;

#define SFPP_GRADIENT_TYPES(WAVEFIELD_INIT, QUADRATURE_INIT, JACOBIAN_INIT)                        \
    std::tuple<RangeTag, WAVEFIELD_INIT, QUADRATURE_INIT, JACOBIAN_INIT>,                          \
        std::tuple<MDRangeTag, WAVEFIELD_INIT, QUADRATURE_INIT, JACOBIAN_INIT>,                    \
        std::tuple<TeamPolicyTag, WAVEFIELD_INIT, QUADRATURE_INIT, JACOBIAN_INIT>,                 \
        std::tuple<TeamPolicyWScratchVTag, WAVEFIELD_INIT, QUADRATURE_INIT, JACOBIAN_INIT>,        \
        std::tuple<TeamPolicyWChunkedScratchVTag, WAVEFIELD_INIT, QUADRATURE_INIT, JACOBIAN_INIT>, \
        std::tuple<TeamPolicyWTiledScratchVTag, WAVEFIELD_INIT, QUADRATURE_INIT, JACOBIAN_INIT>

// Test fixture for gradient operator tests
template <typename TestingTypes>
class GradientTest : public ::testing::Test {
protected:
    using ParallelizationStrategy = std::tuple_element_t<0, TestingTypes>;
    using WavefieldInitializer = std::tuple_element_t<1, TestingTypes>;
    using QuadratureInitializer = std::tuple_element_t<2, TestingTypes>;
    using JacobianInitializer = std::tuple_element_t<3, TestingTypes>;
    using Layout = Kokkos::LayoutRight;  // Fixed layout for tests
    using ExecSpace = Kokkos::DefaultExecutionSpace;

    using WavefieldType = Wavefield<WavefieldInitializer, Layout, ExecSpace>;
    using QuadratureType = Quadrature<QuadratureInitializer, Layout, ExecSpace>;
    using JacobianType = JacobianMatrix2D<JacobianInitializer, Layout, ExecSpace>;
    using GradientType =
        Gradient<ParallelizationStrategy, WavefieldType, QuadratureType, JacobianType>;

    GradientTest()
        : field_(WavefieldInitializer{8, 8, 8, 1}), lprime_(QuadratureInitializer{8}),
          J_(JacobianInitializer{8, 8, 8}) {
        // Allocate reference gradient
        const auto gradient = Gradient(SerialTag{}, field_, lprime_, J_)();
        reference_gradient_ = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gradient);
    }

    void SetUp() override {
        // Compute reference gradient using serial implementation
    }

    WavefieldType field_;
    QuadratureType lprime_;
    JacobianType J_;
    Kokkos::View<float*****, Layout, Kokkos::DefaultHostExecutionSpace> reference_gradient_;
};

// Define the types for typed tests
using GradientTypes =
    ::testing::Types<std::tuple<TeamPolicyWTiledScratchVTag, WavefieldUniformInitializer2D,
                                QuadratureIdentityInitializer, JacobianMatrixRegularInitializer2D>>;

TYPED_TEST_SUITE(GradientTest, GradientTypes);

TYPED_TEST(GradientTest, CompareAgainstSerial) {
    using ParallelizationStrategy = typename TestFixture::ParallelizationStrategy;
    const auto test_gradient =
        Gradient(ParallelizationStrategy{}, this->field_, this->lprime_, this->J_)();

    // Compare dimensions
    ASSERT_EQ(test_gradient.extent(0), this->reference_gradient_.extent(0))
        << "n_elements mismatch";
    ASSERT_EQ(test_gradient.extent(1), this->reference_gradient_.extent(1)) << "nz mismatch";
    ASSERT_EQ(test_gradient.extent(2), this->reference_gradient_.extent(2)) << "nx mismatch";
    ASSERT_EQ(test_gradient.extent(3), this->reference_gradient_.extent(3))
        << "ncomponents mismatch";
    ASSERT_EQ(test_gradient.extent(4), this->reference_gradient_.extent(4))
        << "gradient dims mismatch";

    const auto test_gradient_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, test_gradient);

    // Compare values with tolerance
    const float tolerance = 1e-5f;  // Slightly relaxed tolerance for floating point
    const auto n_elements = test_gradient.extent(0);
    const auto nz = test_gradient.extent(1);
    const auto nx = test_gradient.extent(2);
    const auto ncomponents = test_gradient.extent(3);

    size_t error_count = 0;
    const size_t max_errors_to_show = 5;

    for (size_t e = 0; e < n_elements; ++e) {
        for (size_t iz = 0; iz < nz; ++iz) {
            for (size_t ix = 0; ix < nx; ++ix) {
                for (size_t c = 0; c < ncomponents; ++c) {
                    for (size_t d = 0; d < 2; ++d) {
                        const float test_val = test_gradient_host(e, iz, ix, c, d);
                        const float ref_val = this->reference_gradient_(e, iz, ix, c, d);
                        const float diff = std::abs(test_val - ref_val);

                        if (diff > tolerance) {
                            if (error_count < max_errors_to_show) {
                                std::cout << "Error at (" << e << ", " << iz << ", " << ix << ", "
                                          << c << ", " << d << "): "
                                          << "test = " << test_val << ", reference = " << ref_val
                                          << ", diff = " << diff << std::endl;
                            }
                            error_count++;
                        }
                    }
                }
            }
        }
    }

    EXPECT_EQ(error_count, 0u) << "Found " << error_count
                               << " mismatches between test and reference gradients";
}

// Main function to initialize Kokkos and run tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment());

    int result = RUN_ALL_TESTS();

    return result;
}
