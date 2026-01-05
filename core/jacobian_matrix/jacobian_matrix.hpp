#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {

template <typename Initializer, typename Layout = Kokkos::DefaultExecutionSpace::array_layout,
          typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class JacobianMatrix2D : public Kokkos::View<float*****, Layout, ExecutionSpace> {
public:
    using initializer_type = Initializer;
    using ViewType = Kokkos::View<float*****, Layout, ExecutionSpace>;
    using execution_space = ExecutionSpace;
    using memory_space = typename ViewType::memory_space;
    using layout = Layout;
    JacobianMatrix2D(const Initializer init) {
        init.initialize(static_cast<ViewType&>(*this));

        Kokkos::fence();
    }

    JacobianMatrix2D(const ViewType& other) : ViewType(other) {
    }

    using ViewType::ViewType;

    template <typename E>
    JacobianMatrix2D<initializer_type, layout, E>
    create_mirror_view_and_copy(const E exec_space) const {
        using MirrorType = JacobianMatrix2D<initializer_type, layout, E>;
        const auto view_mirror =
            Kokkos::create_mirror_view_and_copy(exec_space, static_cast<ViewType>(*this));
        return MirrorType(view_mirror);
    }
};

}  // namespace sfpp_playground

#include "initializer/regular.hpp"
