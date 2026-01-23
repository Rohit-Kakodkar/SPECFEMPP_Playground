#pragma once

#include <Kokkos_Core.hpp>
#include <cute/tensor.hpp>

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
    using initializer_type = Initializer;
    using ViewType = Kokkos::View<float**, Layout, ExecutionSpace>;
    using execution_space = ExecutionSpace;
    using memory_space = typename ViewType::memory_space;
    using layout = Layout;

    Quadrature(const ViewType& xi, const ViewType& gamma) : xi_(xi), gamma_(gamma) {
    }

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

    template <typename E>
    Quadrature<initializer_type, layout, E> create_mirror_view_and_copy(const E exec_space) const {
        using MirrorType = Quadrature<initializer_type, layout, E>;
        const auto xi_mirror =
            Kokkos::create_mirror_view_and_copy(exec_space, static_cast<ViewType>(xi_));
        const auto gamma_mirror =
            Kokkos::create_mirror_view_and_copy(exec_space, static_cast<ViewType>(gamma_));
        return MirrorType(xi_mirror, gamma_mirror);
    }

    template <int NGLL>
    auto cute_tensor() const {
        const auto shape = [&]() {
            const auto dim0 = cute::Int<NGLL>{};
            const auto dim1 = cute::Int<NGLL>{};
            return cute::make_shape(dim0, dim1);
        }();
        const auto stride = [&]() {
            const auto stride0 = xi_.stride_0();
            const auto stride1 = xi_.stride_1();
            return cute::make_shape(stride0, stride1);
        }();
        return std::make_pair(cute::make_tensor(xi_.data(), cute::make_layout(shape, stride)),
                              cute::make_tensor(gamma_.data(), cute::make_layout(shape, stride)));
    };

private:
    ViewType xi_;
    ViewType gamma_;
};

}  // namespace sfpp_playground

#include "initializer/identity.hpp"
