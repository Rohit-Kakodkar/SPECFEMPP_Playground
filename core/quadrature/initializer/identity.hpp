#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {

class QuadratureIdentityInitializer {
public:
    QuadratureIdentityInitializer(size_t ngll) : ngll_(ngll) {
    }
    template <typename ViewType>
    void initialize(ViewType& xi, ViewType& gamma) const {
        static_assert(ViewType::rank() == 2, "QuadratureIdentityInitializer requires rank-2 views");
        using ExecSpace = typename ViewType::execution_space;
        xi = ViewType("xi", ngll_, ngll_);
        gamma = ViewType("gamma", ngll_, ngll_);
        Kokkos::parallel_for(
            "InitializeIdentityQuadrature", Kokkos::RangePolicy<ExecSpace>(0, ngll_),
            KOKKOS_CLASS_LAMBDA(const size_t i) {
                for (size_t j = 0; j < ngll_; ++j) {
                    xi(i, j) = (i == j) ? 1.0f : 0.0f;
                    gamma(i, j) = (i == j) ? 1.0f : 0.0f;
                }
            });
    }

private:
    size_t ngll_;
};

}  // namespace sfpp_playground
