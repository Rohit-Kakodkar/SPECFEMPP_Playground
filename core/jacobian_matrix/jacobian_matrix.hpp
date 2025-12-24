#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {

template <typename Initializer, typename Layout = Kokkos::DefaultExecutionSpace::array_layout,
          typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class JacobianMatrix2D : public Kokkos::View<float*****, Layout, ExecutionSpace> {
public:
    using ViewType = Kokkos::View<float*****, Layout, ExecutionSpace>;
    using execution_space = ExecutionSpace;
    using memory_space = typename ViewType::memory_space;
    using layout = Layout;
    JacobianMatrix2D(const Initializer init) {
        init.initialize(static_cast<ViewType&>(*this));

        Kokkos::fence();
    }

    using ViewType::ViewType;
};

}  // namespace sfpp_playground

#include "initializer/regular.hpp"
