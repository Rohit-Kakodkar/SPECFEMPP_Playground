#include <SFPP_playground.hpp>

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        using namespace sfpp_playground;

        constexpr int n_elements = 1024;
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
        auto gradient_serial = Gradient(SerialTag{}, field, lprime, J)();
        auto gradient_mdrange = Gradient(MDRangeTag{}, field, lprime, J)();
        auto gradient_team = Gradient(TeamPolicyTag{}, field, lprime, J)();
        auto gradient_team_scratch = Gradient(TeamPolicyWScratchVTag{}, field, lprime, J)();
        auto gradient_team_chunked_scratch =
            Gradient(TeamPolicyWChunkedScratchVTag{}, field, lprime, J)();
    }
    Kokkos::finalize();
    return 0;
}