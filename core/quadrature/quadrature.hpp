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
    using execution_space = ExecutionSpace;
    using memory_space = typename ViewType::memory_space;
    using layout = Layout;

    KOKKOS_INLINE_FUNCTION
    auto xi() const {
        return xi_;
    }

    KOKKOS_INLINE_FUNCTION
    auto gamma() const {
        return gamma_;
    }

    KOKKOS_INLINE_FUNCTION
    float& xi(const size_t i, const size_t j) const {
        return xi_(i, j);
    }

    KOKKOS_INLINE_FUNCTION
    float& gamma(const size_t i, const size_t j) const {
        return gamma_(i, j);
    }

private:
    ViewType xi_;
    ViewType gamma_;
};

}  // namespace sfpp_playground

#include "initializer/identity.hpp"
