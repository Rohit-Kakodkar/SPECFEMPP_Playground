#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {

class JacobianMatrixRegularInitializer2D {
public:
    JacobianMatrixRegularInitializer2D(size_t n_elements, int nz, size_t nx)
        : n_elements_(n_elements), nz_(nz), nx_(nx) {
    }

    template <typename ViewType>
    void initialize(ViewType& J) const {
        static_assert(ViewType::rank() == 5,
                      "JacobianMatrixRegularInitializer2D requires a rank-5 view");
        using ExecSpace = typename ViewType::execution_space;
        J = ViewType("JacobianMatrix", n_elements_, nz_, nx_, 2, 2);
        Kokkos::parallel_for(
            "InitializeRegularJacobianMatrix",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>({0, 0, 0}, {n_elements_, nz_, nx_}),
            KOKKOS_CLASS_LAMBDA(const size_t e, const size_t iz, const size_t ix) {
                J(e, iz, ix, 0, 0) = 1.0f;  // dxi/dx
                J(e, iz, ix, 0, 1) = 0.0f;  // dxi/dz
                J(e, iz, ix, 1, 0) = 0.0f;  // dgamma/dx
                J(e, iz, ix, 1, 1) = 1.0f;  // dgamma/dz
            });
    }

private:
    size_t n_elements_;
    size_t nz_;
    size_t nx_;
};

}  // namespace sfpp_playground
