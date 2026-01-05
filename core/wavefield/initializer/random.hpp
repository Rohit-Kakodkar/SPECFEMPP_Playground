#pragma once
#include <Kokkos_Core.hpp>
#include <cstdlib>

namespace sfpp_playground {
class WavefieldRandomInitializer2D {
public:
    WavefieldRandomInitializer2D(const size_t nelements, const size_t nz, const size_t nx,
                                 const size_t ncomponents)
        : nelements_(nelements), nz_(nz), nx_(nx), ncomponents_(ncomponents) {
    }
    template <typename ViewType>
    void initialize(ViewType& view) const {
        view = ViewType("RandomInitializedView", nelements_, nz_, nx_, ncomponents_, ncomponents_);
        using HostSpace = Kokkos::DefaultHostExecutionSpace;

        const auto host_view = Kokkos::create_mirror_view_and_copy(HostSpace(), view);

        std::srand(0);  // For reproducibility

        Kokkos::parallel_for(
            "InitializeRandomView",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>, HostSpace>({0, 0, 0, 0},
                                                              {nelements_, nz_, nx_, ncomponents_}),
            [=, *this](const size_t e, const size_t iz, const size_t ix, const size_t ic) {
                host_view(e, iz, ix, ic) = std::rand();
            });

        Kokkos::fence();
        Kokkos::deep_copy(view, host_view);
    }

private:
    size_t nelements_;
    size_t nz_;
    size_t nx_;
    size_t ncomponents_;
};
}  // namespace sfpp_playground
