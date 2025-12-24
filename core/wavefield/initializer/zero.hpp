#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {
class WavefieldZeroInitializer2D {
public:
    WavefieldZeroInitializer2D(const size_t nelements, const size_t nz, const size_t nx,
                               const size_t ncomponents)
        : nelements_(nelements), nz_(nz), nx_(nx), ncomponents_(ncomponents) {
    }
    template <typename ViewType>
    void initialize(ViewType& view) const {
        view = ViewType("ZeroInitializedView", nelements_, nz_, nx_, ncomponents_, ncomponents_);
        using ExecSpace = typename ViewType::execution_space;
        Kokkos::parallel_for(
            "InitializeZeroView",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>({0, 0, 0, 0},
                                                              {nelements_, nz_, nx_, ncomponents_}),
            KOKKOS_CLASS_LAMBDA(const size_t e, const size_t iz, const size_t ix, const size_t ic) {
                view(e, iz, ix, ic) = 0.0f;
            });
    }

private:
    size_t nelements_;
    size_t nz_;
    size_t nx_;
    size_t ncomponents_;
};
}  // namespace sfpp_playground
