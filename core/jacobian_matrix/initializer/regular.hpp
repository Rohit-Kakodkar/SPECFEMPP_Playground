#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {

class JacobianMatrixRegularInitializer2D {
public:
    JacobianMatrixRegularInitializer2D(int n_elements, int nz, int nx)
        : n_elements_(n_elements), nz_(nz), nx_(nx) {
    }

    template <typename ViewType>
    void initialize(ViewType& J) const {
        static_assert(ViewType::rank() == 5,
                      "JacobianMatrixRegularInitializer2D requires a rank-5 view");
        J = ViewType("JacobianMatrix", n_elements_, nz_, nx_, 2, 2);
        Kokkos::parallel_for(
            "InitializeRegularJacobianMatrix",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n_elements_, nz_, nx_}),
            KOKKOS_CLASS_LAMBDA(const int e, const int iz, const int ix) {
                J(e, iz, ix, 0, 0) = 1.0f;  // dxi/dx
                J(e, iz, ix, 0, 1) = 0.0f;  // dxi/dz
                J(e, iz, ix, 1, 0) = 0.0f;  // dgamma/dx
                J(e, iz, ix, 1, 1) = 1.0f;  // dgamma/dz
            });
    }

private:
    int n_elements_;
    int nz_;
    int nx_;
};

}  // namespace sfpp_playground