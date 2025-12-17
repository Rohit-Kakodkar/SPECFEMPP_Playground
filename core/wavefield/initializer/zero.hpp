#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {
class WavefieldZeroInitializer2D {
public:
    WavefieldZeroInitializer2D(const int nelements, const int nz, const int nx,
                               const int ncomponents)
        : nelements_(nelements), nz_(nz), nx_(nx), ncomponents_(ncomponents) {
    }
    template <typename ViewType>
    void initialize(ViewType& view) const {
        view = ViewType("ZeroInitializedView", nelements_, nz_, nx_, ncomponents_, ncomponents_);
        Kokkos::parallel_for(
            "InitializeZeroView",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0},
                                                   {nelements_, nz_, nx_, ncomponents_}),
            KOKKOS_CLASS_LAMBDA(const int e, const int iz, const int ix, const int ic) {
                view(e, iz, ix, ic) = 0.0f;
            });
    }

private:
    int nelements_;
    int nz_;
    int nx_;
    int ncomponents_;
};
}  // namespace sfpp_playground