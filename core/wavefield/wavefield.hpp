#pragma once

#include <Kokkos_Core.hpp>
#include <cute/tensor.hpp>

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

    template <int NGLL, int NCOMP>
    auto cute_tensor() const {
        static_assert(std::is_same_v<Layout, Kokkos::LayoutLeft>,
                      "Currently only LayoutLeft is supported for cute_tensor");
        const auto shape = [&]() {
            const auto dim0 = this->extent(0);
            const auto dim1 = cute::Int<NGLL>{};
            const auto dim2 = cute::Int<NGLL>{};
            const auto dim3 = cute::Int<NCOMP>{};
            return cute::make_shape(dim0, dim1, dim2, dim3);
        }();
        const auto stride = [&]() {
            const auto stride0 = cute::Int<1>{};
            const auto stride1 = this->stride_1();
            const auto stride2 = this->stride_2();
            const auto stride3 = this->stride_3();
            return cute::make_shape(stride0, stride1, stride2, stride3);
        }();
        return cute::make_tensor(cute::make_gmem_ptr(this->data()),
                                 cute::make_layout(shape, stride));
    };
};

}  // namespace sfpp_playground

#include "initializer/element.hpp"
#include "initializer/random.hpp"
#include "initializer/uniform.hpp"
#include "initializer/zero.hpp"
