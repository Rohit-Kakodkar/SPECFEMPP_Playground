#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {

template <typename Initializer, typename Layout = Kokkos::DefaultExecutionSpace::array_layout,
          typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class Quadrature {
public:
    Quadrature(const Initializer init) {
        init.initialize(xi_, gamma_);

        Kokkos::fence();
    }
    Quadrature() = default;
    using ViewType = Kokkos::View<float**, Layout, ExecutionSpace>;

    KOKKOS_INLINE_FUNCTION
    float& xi(const int i, const int j) const {
        return xi_(i, j);
    }

    KOKKOS_INLINE_FUNCTION
    float& gamma(const int i, const int j) const {
        return gamma_(i, j);
    }

private:
    ViewType xi_;
    ViewType gamma_;
};

}  // namespace sfpp_playground

#include "initializer/identity.hpp"
