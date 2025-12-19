#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {
namespace impl {
template <typename FieldView, typename Quadrature, typename JacobianMatrixType>
class GradientBase {
public:
    GradientBase(const FieldView& field, const Quadrature& lprime, const JacobianMatrixType& J)
        : field_(field), lprime_(lprime), J_(J),
          gradient_("gradient", field.extent(0), field.extent(1), field.extent(2), 1, 2),
          nx_(field.extent(2)), nz_(field.extent(1)), n_elements_(field.extent(0)) {
    }

    using memory_space = FieldView::memory_space;
    using execution_space = typename FieldView::execution_space;
    using layout = FieldView::array_layout;
    using T = typename FieldView::value_type;
    using ReturnType = Kokkos::View<T*****, layout, memory_space>;

    FieldView field_;
    Quadrature lprime_;
    JacobianMatrixType J_;
    ReturnType gradient_;

    constexpr static int ngll_ = 5;
    constexpr static int ncomponents_ = 1;
    int nx_;
    int nz_;
    int n_elements_;
};
}  // namespace impl

template <typename ParallelizationStrategy, typename FieldView, typename Quadrature,
          typename JacobianMatrixType>
class Gradient;

// CTAD guide
template <typename ParallelizationStrategy, typename FieldView, typename Quadrature,
          typename JacobianMatrixType>
Gradient(const ParallelizationStrategy /*unused*/, const FieldView& field, const Quadrature& lprime,
         const JacobianMatrixType& J)
    -> Gradient<ParallelizationStrategy, FieldView, Quadrature, JacobianMatrixType>;

}  // namespace sfpp_playground

#include "implementation/md_range.hpp"
#include "implementation/serial.hpp"
#include "implementation/team_policy.hpp"
#include "implementation/team_policy_w_chunked_scratch_view_tag.hpp"
#include "implementation/team_policy_w_scratch_view.hpp"
