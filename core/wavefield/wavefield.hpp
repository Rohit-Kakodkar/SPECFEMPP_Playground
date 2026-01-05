#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {

template <typename Initializer, typename Layout = Kokkos::DefaultExecutionSpace::array_layout,
          typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class Wavefield : public Kokkos::View<float****, Layout, ExecutionSpace> {
public:
    using initializer_type = Initializer;
    using ViewType = Kokkos::View<float****, Layout, ExecutionSpace>;
    using execution_space = ExecutionSpace;
    using memory_space = typename ViewType::memory_space;
    using layout = Layout;
    Wavefield(const Initializer init) {
        init.initialize(static_cast<ViewType&>(*this));

        Kokkos::fence();
    }

    Wavefield(const ViewType& other) : ViewType(other) {
    }

    using ViewType::ViewType;

    template <typename E>
    Wavefield<initializer_type, layout, E> create_mirror_view_and_copy(const E exec_space) const {
        using MirrorType = Wavefield<initializer_type, layout, E>;
        const Kokkos::View<float****, layout, E> view_mirror =
            Kokkos::create_mirror_view_and_copy(exec_space, static_cast<ViewType>(*this));

        return MirrorType(view_mirror);
    }
};

}  // namespace sfpp_playground

#include "initializer/random.hpp"
#include "initializer/uniform.hpp"
#include "initializer/zero.hpp"
