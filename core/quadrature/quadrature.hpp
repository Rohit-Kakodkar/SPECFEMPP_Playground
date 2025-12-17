#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {

template <typename Initializer>
class Quadrature {
public:
    Quadrature(const Initializer init) {
        init.initialize(xi_, gamma_);

        Kokkos::fence();
    }
    Quadrature() = default;
    using ViewType = Kokkos::View<float**, Kokkos::DefaultExecutionSpace>;

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
