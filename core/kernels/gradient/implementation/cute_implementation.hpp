#pragma once

#include <Kokkos_Core.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/underscore.hpp>

using namespace cute;

namespace sfpp_playground {

struct CuteImplementationTag {};

template <int NCOMP, typename CtaTiler, typename FTensor, typename FxSmemLayout,
          typename FzSmemLayout, typename QTensor, typename XiSmemLayout, typename GammaSmemLayout,
          typename JTensor, typename GTensor>
__global__ void
compute_gradient_cute_kernel(const CtaTiler cta_tiler, const FTensor field,
                             FxSmemLayout fx_smem_layout, FzSmemLayout fz_smem_layout,
                             const QTensor xi, const QTensor gamma, XiSmemLayout xi_smem_layout,
                             GammaSmemLayout gamma_smem_layout, const JTensor J, GTensor gradient) {
    auto cta_coord = make_coord(blockIdx.x, _, _, _);

    constexpr size_t bN = size<0>(cta_tiler);
    constexpr size_t bNz = size<1>(cta_tiler);
    constexpr size_t bNx = size<2>(cta_tiler);
    constexpr size_t bNl = size<3>(cta_tiler);
    constexpr size_t ncomponents = NCOMP;

    auto thread_tiler = make_shape(Int<1>{}, Int<bNz>{}, Int<bNx>{}, Int<bNl>{});

    auto thread_coord = make_coord(threadIdx.x, _, _, _);

    auto gFx = local_tile(field, select<0, 1, 3>(cta_tiler),
                          select<0, 1, 3>(cta_coord));  // (bN, bNz, bNl, NTilez, NTilek, ncomp)
    auto gFz = local_tile(field, select<0, 3, 2>(cta_tiler),
                          select<0, 3, 2>(cta_coord));  // (bN, bNl, bNx, NTilek, NTilex, ncomp)
    auto gJ = local_tile(J, select<0, 1, 2>(cta_tiler),
                         select<0, 1, 2>(cta_coord));  // (bN, bNz, bNx, NTilez, NTilex, 2, 2)
    auto gG = local_tile(gradient, select<0, 1, 2>(cta_tiler),
                         select<0, 1, 2>(cta_coord));  // (bN, bNz, bNx, NTilez, NTilex, ncomp, 2)
    auto gXi = local_tile(xi, select<2, 3>(cta_tiler),
                          select<2, 3>(cta_coord));  // (bNx, bNl, NTilex, NTilek)
    auto gGamma = local_tile(gamma, select<1, 3>(cta_tiler),
                             select<1, 3>(cta_coord));  // (bNz, bNl, NTilez, NTilek)

    constexpr auto Ntilez = size<3>(gG);
    constexpr auto Ntilex = size<4>(gG);
    constexpr auto Ntilek = size<4>(gFx);

    // // Make sure dimensions match

    // gFx dimensions
    static_assert(size<0>(gFx) == bN, "gFx dimension 0 mismatch");
    static_assert(size<1>(gFx) == bNz, "gFx dimension 1 mismatch");
    static_assert(size<2>(gFx) == bNl, "gFx dimension 2 mismatch");
    static_assert(size<3>(gFx) == Ntilez, "gFx dimension 3 mismatch");
    static_assert(size<4>(gFx) == Ntilek, "gFx dimension 4 mismatch");
    static_assert(size<5>(gFx) == ncomponents, "gFx dimension 5 mismatch");

    // gFz dimensions
    static_assert(size<0>(gFz) == bN, "gFz dimension 0 mismatch");
    static_assert(size<1>(gFz) == bNl, "gFz dimension 1 mismatch");
    static_assert(size<2>(gFz) == bNx, "gFz dimension 2 mismatch");
    static_assert(size<3>(gFz) == Ntilek, "gFz dimension 3 mismatch");
    static_assert(size<4>(gFz) == Ntilex, "gFz dimension 4 mismatch");
    static_assert(size<5>(gFz) == ncomponents, "gFz dimension 5 mismatch");

    // gJ dimensions
    static_assert(size<0>(gJ) == bN, "gJ dimension 0 mismatch");
    static_assert(size<1>(gJ) == bNz, "gJ dimension 1 mismatch");
    static_assert(size<2>(gJ) == bNx, "gJ dimension 2 mismatch");
    static_assert(size<3>(gJ) == Ntilez, "gJ dimension 3 mismatch");
    static_assert(size<4>(gJ) == Ntilex, "gJ dimension 4 mismatch");
    static_assert(size<5>(gJ) == 2, "gJ dimension 5 mismatch");
    static_assert(size<6>(gJ) == 2, "gJ dimension 6 mismatch");

    // gG dimensions
    static_assert(size<0>(gG) == bN, "gG dimension 0 mismatch");
    static_assert(size<1>(gG) == bNz, "gG dimension 1 mismatch");
    static_assert(size<2>(gG) == bNx, "gG dimension 2 mismatch");
    static_assert(size<3>(gG) == Ntilez, "gG dimension 3 mismatch");
    static_assert(size<4>(gG) == Ntilex, "gG dimension 4 mismatch");
    static_assert(size<5>(gG) == ncomponents, "gG dimension 5 mismatch");
    static_assert(size<6>(gG) == 2, "gG dimension 6 mismatch");

    // gXi dimensions
    static_assert(size<0>(gXi) == bNx, "gXi dimension 0 mismatch");
    static_assert(size<1>(gXi) == bNl, "gXi dimension 1 mismatch");
    static_assert(size<2>(gXi) == Ntilex, "gXi dimension 2 mismatch");
    static_assert(size<3>(gXi) == Ntilek, "gXi dimension 3 mismatch");

    // gGamma dimensions
    static_assert(size<0>(gGamma) == bNz, "gGamma dimension 0 mismatch");
    static_assert(size<1>(gGamma) == bNl, "gGamma dimension 1 mismatch");
    static_assert(size<2>(gGamma) == Ntilez, "gGamma dimension 2 mismatch");
    static_assert(size<3>(gGamma) == Ntilek, "gGamma dimension 3 mismatch");

    __shared__ typename FTensor::value_type field_x_smem[cosize_v<FxSmemLayout>];
    __shared__ typename FTensor::value_type field_z_smem[cosize_v<FzSmemLayout>];
    __shared__ typename QTensor::value_type lprime_xi_smem[cosize_v<XiSmemLayout>];
    __shared__ typename QTensor::value_type lprime_gamma_smem[cosize_v<GammaSmemLayout>];

    Tensor sFx = make_tensor(make_smem_ptr(field_x_smem), fx_smem_layout);
    Tensor sFz = make_tensor(make_smem_ptr(field_z_smem), fz_smem_layout);
    Tensor sXi = make_tensor(make_smem_ptr(lprime_xi_smem), xi_smem_layout);
    Tensor sGamma = make_tensor(make_smem_ptr(lprime_gamma_smem), gamma_smem_layout);

    const size_t iz = threadIdx.y;
    const size_t ix = threadIdx.z;

    for (size_t tile_k = 0; tile_k < Ntilek; ++tile_k) {
        for (size_t tile_z = 0; tile_z < Ntilez; ++tile_z) {
            for (size_t c = 0; c < ncomponents; ++c) {
                for (size_t l = 0; l < bNl; ++l) {
                    sFx(threadIdx.x, iz, l, c) = gFx(threadIdx.x, iz, l, tile_z, tile_k, c);
                    sGamma(iz, l) = gGamma(iz, l, tile_z, tile_k);
                }
            }

            for (size_t tile_x = 0; tile_x < Ntilex; ++tile_x) {
                for (size_t c = 0; c < ncomponents; ++c) {
                    for (size_t l = 0; l < bNl; ++l) {
                        sFz(threadIdx.x, l, ix, c) = gFz(threadIdx.x, l, ix, tile_k, tile_x, c);
                        sXi(ix, l) = gXi(ix, l, tile_x, tile_k);
                    }
                }

                __syncthreads();
                // Compute gradients
                typename GTensor::value_type grad_x[ncomponents] = {
                    static_cast<typename GTensor::value_type>(0)};
                typename GTensor::value_type grad_z[ncomponents] = {
                    static_cast<typename GTensor::value_type>(0)};
                for (size_t l = 0; l < bNl; ++l) {
                    for (size_t c = 0; c < ncomponents; ++c) {
                        grad_x[c] += sFx(threadIdx.x, iz, l, c) * sXi(ix, l);
                        grad_z[c] += sFz(threadIdx.x, l, ix, c) * sGamma(iz, l);
                    }
                }

                for (size_t c = 0; c < ncomponents; ++c) {
                    gG(threadIdx.x, iz, ix, tile_z, tile_x, c, 0) +=
                        gJ(threadIdx.x, iz, ix, tile_z, tile_x, 0, 0) * grad_x[c] +
                        gJ(threadIdx.x, iz, ix, tile_z, tile_x, 0, 1) * grad_z[c];
                    gG(threadIdx.x, iz, ix, tile_z, tile_x, c, 1) +=
                        gJ(threadIdx.x, iz, ix, tile_z, tile_x, 1, 0) * grad_x[c] +
                        gJ(threadIdx.x, iz, ix, tile_z, tile_x, 1, 1) * grad_z[c];
                }

                __syncthreads();
            }
        }
    }
}

template <typename FieldView, typename Quadrature, typename JacobianMatrixType>
class Gradient<CuteImplementationTag, FieldView, Quadrature, JacobianMatrixType>
    : private impl::GradientBase<FieldView, Quadrature, JacobianMatrixType> {
public:
    using Base = impl::GradientBase<FieldView, Quadrature, JacobianMatrixType>;
    using typename Base::ReturnType;
    using typename Base::T;

    Gradient(const CuteImplementationTag /*unused*/, const FieldView& field,
             const Quadrature& lprime, const JacobianMatrixType& J)
        : Base(field, lprime, J) {
    }

    static std::string name() {
        return "CuteImplementationTag";
    }

    using Base::Base;

    ReturnType operator()() const {
        // CUTE-based implementation placeholder
        const size_t n_elements = this->n_elements_;
        const size_t nx = this->nx_;
        const size_t nz = this->nz_;

        const auto f = this->field_.template cute_tensor<Base::ngll_, Base::ncomponents_>();
        const auto [xi, gamma] = this->lprime_.template cute_tensor<Base::ngll_>();
        const auto j = this->J_.template cute_tensor<Base::ngll_>();
        auto g = make_tensor(
            this->gradient_.data(),
            make_layout(make_shape(n_elements, Int<Base::ngll_>{}, Int<Base::ngll_>{},
                                   Int<Base::ncomponents_>{}, Int<2>{}),
                        make_shape(this->gradient_.stride_0(), this->gradient_.stride_1(),
                                   this->gradient_.stride_2(), this->gradient_.stride_3(),
                                   this->gradient_.stride_4())));

        constexpr size_t bN = 32;
        constexpr size_t bNx = 4;
        constexpr size_t bNz = 2;
        constexpr size_t bNl = 8;

        const auto cta_tiler = make_shape(Int<bN>{}, Int<bNz>{}, Int<bNx>{}, Int<bNl>{});

        auto fx_smem_layout = make_layout(
            make_shape(Int<bN>{}, Int<bNz>{}, Int<bNl>{}, Int<Base::ncomponents_>{}), LayoutLeft{});
        auto fz_smem_layout = make_layout(
            make_shape(Int<bN>{}, Int<bNl>{}, Int<bNx>{}, Int<Base::ncomponents_>{}), LayoutLeft{});

        auto Xi_smem_layout = make_layout(make_shape(Int<bNx>{}, Int<bNl>{}), LayoutLeft{});
        auto Gamma_smem_layout = make_layout(make_shape(Int<bNz>{}, Int<bNl>{}), LayoutLeft{});

        dim3 grid((n_elements + bN - 1) / bN);
        dim3 block(bN, bNz, bNx);

        sfpp_playground::compute_gradient_cute_kernel<Base::ncomponents_>
            <<<grid, block>>>(cta_tiler, f, fx_smem_layout, fz_smem_layout, xi, gamma,
                              Xi_smem_layout, Gamma_smem_layout, j, g);

        Kokkos::fence();

        return this->gradient_;
    }

private:
    // Private helper methods for CUTE implementation can be added here
};
}  // namespace sfpp_playground
