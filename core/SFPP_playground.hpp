#pragma once

#include "jacobian_matrix/jacobian_matrix.hpp"
#include "kernels/kernels.hpp"
#include "quadrature/quadrature.hpp"
#include "wavefield/wavefield.hpp"

namespace sfpp_playground {
template <typename ExecutionSpace, typename View>
auto create_mirror_view_and_copy(const ExecutionSpace exec_space, const View& source) {
    return source.create_mirror_view_and_copy(exec_space);
}
}  // namespace sfpp_playground
