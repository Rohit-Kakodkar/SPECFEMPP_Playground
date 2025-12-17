#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {

template <typename Initializer>
class JacobianMatrix2D : public Kokkos::View<float*****, Kokkos::DefaultExecutionSpace> {
public:
    using ViewType = Kokkos::View<float*****, Kokkos::DefaultExecutionSpace>;
    JacobianMatrix2D(const Initializer init) {
        init.initialize(static_cast<ViewType&>(*this));

        Kokkos::fence();
    }

    using ViewType::ViewType;
};

}  // namespace sfpp_playground

#include "initializer/regular.hpp"