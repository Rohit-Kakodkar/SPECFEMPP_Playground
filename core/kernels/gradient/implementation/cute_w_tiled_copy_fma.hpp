#pragma once

#include <Kokkos_Core.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/underscore.hpp>

using namespace cute;

namespace sfpp_playground {

struct CuteCopyFMATag {};

namespace detail {

template <typename TupleType, typename I0>
constexpr auto reorder_tuple(TupleType const& tp, I0) {
    if constexpr (is_tuple<I0>::value) {
        return reorder_tuple(tp, I0{});
    } else {
        return get<I0::value>(tp);
    }
}

template <typename TupleType, typename... Is>
constexpr auto reorder_tuple(TupleType const& tp, Step<Is...>) {
    return make_tuple(reorder_tuple(tp, Is{})...);
}

template <typename LayoutType, typename... Is>
constexpr auto reorder_layout(LayoutType const& layout, Step<Is...>) {
    return make_layout(reorder_tuple(layout.shape(), Step<Is...>{}),
                       reorder_tuple(layout.stride(), Step<Is...>{}));
}

template <typename Tensor, typename... Is>
constexpr auto reorder_tensor(const Tensor& tensor, Step<Is...>) {
    return make_tensor(tensor.data(), reorder_layout(tensor.layout(), Step<Is...>{}));
}

template <typename Tensor, size_t I0, size_t... Is>
constexpr auto group(const Tensor& tensor, index_sequence<I0, Is...>) {
    return reorder_tensor(tensor, Step<Int<I0>, Step<Int<Is>...>>{});
}

template <typename Tensor>
constexpr auto group(const Tensor& tensor) {
    return group(tensor, make_index_sequence<decltype(rank(tensor))::value>{});
}

}  // namespace detail

template <typename CtaTiler, typename FTensor, typename FxSmemLayout, typename FzSmemLayout,
          typename QTensor, typename XiSmemLayout, typename GammaSmemLayout, typename JTensor,
          typename GTensor, typename CopyAtom, typename MMAGx, typename MMAGz>
__global__ static void compute_gradient_cute_kernel_w_tiled_copy(
    const CtaTiler cta_tiler, const FTensor field, FxSmemLayout fx_smem_layout,
    FzSmemLayout fz_smem_layout, const QTensor xi, const QTensor gamma, XiSmemLayout xi_smem_layout,
    GammaSmemLayout gamma_smem_layout, const JTensor J, GTensor gradient, CopyAtom tiled_copy,
    MMAGx mmaGx, MMAGz mmaGz) {
    auto cta_coord = make_coord(blockIdx.x, _, _, _, 0, 0);

    constexpr auto bN = size<0>(cta_tiler);
    constexpr auto bNz = size<1>(cta_tiler);
    constexpr auto bNx = size<2>(cta_tiler);
    constexpr auto bNl = size<3>(cta_tiler);
    constexpr auto ncomponents = size<4>(cta_tiler);

    auto j_mod = make_tensor(J.data(), make_layout(cute::tuple_cat(J.shape(), Shape<_1>{}),
                                                   cute::tuple_cat(J.stride(), Shape<_0>{})));

    auto gFx = local_tile(field, select<0, 1, 3, 4>(cta_tiler),
                          select<0, 1, 3, 4>(cta_coord));  // (bN, bNz, bNl, ncomp, NTilez, NTilek)
    auto gFz = local_tile(field, select<0, 3, 2, 4>(cta_tiler),
                          select<0, 3, 2, 4>(cta_coord));  // (bN, bNl, bNx, ncomp, NTilek, NTilex)
    auto gJ = local_tile(
        j_mod, select<0, 1, 2, 5, 5, 4>(cta_tiler),
        select<0, 1, 2, 5, 5, 4>(cta_coord));  // (bN, bNz, bNx, 2, 2, ncomp, NTilez, NTilex)
    auto gG =
        local_tile(gradient, select<0, 1, 2, 4, 5>(cta_tiler),
                   select<0, 1, 2, 4, 5>(cta_coord));  // (bN, bNz, bNx, ncomp, 2, NTilez, NTilex)

    constexpr auto NTilez = size<4>(gFx);
    constexpr auto NTilek = size<4>(gFz);
    constexpr auto NTilex = size<5>(gFz);

    static_assert(size<0>(gFx) == bN && size<1>(gFx) == bNz && size<2>(gFx) == bNl &&
                      size<3>(gFx) == ncomponents && size<4>(gFx) == NTilez &&
                      size<5>(gFx) == NTilek,
                  "Unexpected gFx shape");
    static_assert(size<0>(gFz) == bN && size<1>(gFz) == bNl && size<2>(gFz) == bNx &&
                      size<3>(gFz) == ncomponents && size<4>(gFz) == NTilek &&
                      size<5>(gFz) == NTilex,
                  "Unexpected gFz shape");
    static_assert(size<0>(gJ) == bN && size<1>(gJ) == bNz && size<2>(gJ) == bNx &&
                      size<3>(gJ) == 2 && size<4>(gJ) == 2 && size<5>(gJ) == ncomponents &&
                      size<6>(gJ) == NTilez && size<7>(gJ) == NTilex,
                  "Unexpected gJ shape");
    static_assert(size<0>(gG) == bN && size<1>(gG) == bNz && size<2>(gG) == bNx &&
                      size<3>(gG) == ncomponents && size<4>(gG) == 2 && size<5>(gG) == NTilez &&
                      size<6>(gG) == NTilex,
                  "Unexpected gG shape");

    __shared__ typename FTensor::value_type field_x_smem[cosize_v<FxSmemLayout>];
    __shared__ typename FTensor::value_type field_z_smem[cosize_v<FzSmemLayout>];
    __shared__ typename QTensor::value_type lprime_xi_smem[cosize_v<XiSmemLayout>];
    __shared__ typename QTensor::value_type lprime_gamma_smem[cosize_v<GammaSmemLayout>];

    Tensor sFx = make_tensor(make_smem_ptr(field_x_smem), fx_smem_layout);
    Tensor sFz = make_tensor(make_smem_ptr(field_z_smem), fz_smem_layout);
    Tensor sXi = make_tensor(make_smem_ptr(lprime_xi_smem), xi_smem_layout);
    Tensor sGamma = make_tensor(make_smem_ptr(lprime_gamma_smem), gamma_smem_layout);

    const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    // Copy gFx
    auto cgFx = detail::reorder_tensor(gFx, Step<_0, Step<_1, _2, _3>, _4, _5>{});
    auto csFx = detail::reorder_tensor(sFx, Step<_0, Step<_1, _2, _3>>{});
    auto thr_fx = tiled_copy.get_slice(tid);
    auto tFxgFx = thr_fx.partition_S(cgFx);
    auto tFxsFx = thr_fx.partition_D(csFx);

    // // Copy gFz
    auto cgFz = detail::reorder_tensor(gFz, Step<_0, Step<_1, _2, _3>, _4, _5>{});
    auto csFz = detail::reorder_tensor(sFz, Step<_0, Step<_1, _2, _3>>{});
    auto thr_fz = tiled_copy.get_slice(tid);
    auto tFzgFz = thr_fz.partition_S(cgFz);
    auto tFzsFz = thr_fz.partition_D(csFz);

    auto tFxrFx = make_fragment_like(tFxsFx);
    auto tFzrFz = make_fragment_like(tFzsFz);
    copy(tiled_copy, tFxgFx(_, _, _, 0, 0), tFxrFx);
    copy(tiled_copy, tFzgFz(_, _, _, 0, 0), tFzrFz);

    copy(xi, sXi);
    copy(gamma, sGamma);

    auto psXi = local_tile(sXi, select<3, 2>(cta_tiler),
                           select<3, 2>(cta_coord));  // (bNl, bNx, NTilek, NTilex)
    auto psGamma = local_tile(sGamma, select<1, 3>(cta_tiler),
                              select<1, 3>(cta_coord));  // (bNz, bNl, NTilez, NTilek)

    // Partition tensors for gradient evaluation
    auto esFx =
        detail::reorder_tensor(sFx, Step<Step<_0, _1, _3>, _2>{});  // ((bN, bNz, ncomp), bNl)
    auto esFz =
        detail::reorder_tensor(sFz, Step<Step<_0, _2, _3>, _1>{});     // ((bN, bNx, ncomp), bNl)
    auto esXi = detail::reorder_tensor(psXi, Step<_1, _0, _3, _2>{});  // (bNx, bNl, NTilex, NTilek)
    auto esGamma =
        detail::reorder_tensor(psGamma, Step<_0, _1, _2, _3>{});  // (bNz, bNl, NTilez, NTilek)
    auto egGx = detail::reorder_tensor(
        gG,
        Step<Step<_0, _1, _3>, _2, _4, _5, _6>{});  // ((bN, bNz, ncomp), bNx, 2, NTilez, NTilex)
    auto egGz = detail::reorder_tensor(
        gG,
        Step<_1, Step<_0, _2, _3>, _4, _5, _6>{});  // (bNz, (bN, bNx, ncomp), 2, NTilez, NTilex)
    auto egJx =
        detail::reorder_tensor(gJ, Step<Step<_0, _1, _5>, _2, _3, _4, _6,
                                        _7>{});  // ((bN, bNz, ncomp), bNx, 2, 2, NTilez, NTilex)
    auto egJz =
        detail::reorder_tensor(gJ, Step<_1, Step<_0, _2, _5>, _3, _4, _6,
                                        _7>{});  // (bNz, (bN, bNx, ncomp), 2, 2, NTilez, NTilex)

    const int tid_fx = threadIdx.x + static_cast<int>(bN) * threadIdx.y +
                       static_cast<int>(bN * bNz / 2) * threadIdx.z;
    const int tid_fz = threadIdx.x + static_cast<int>(bN) * threadIdx.z +
                       static_cast<int>(bN * bNx / 2) * threadIdx.y;

    auto thr_mma_gx = mmaGx.get_slice(tid_fx);

    auto tGxsFx = thr_mma_gx.partition_A(esFx);
    auto tGxsGamma = thr_mma_gx.partition_B(esGamma);
    auto tGxgGx = thr_mma_gx.partition_C(egGx);
    auto tJxgGx = thr_mma_gx.partition_C(egJx);

    auto thr_mma_gz = mmaGz.get_slice(tid_fz);

    auto tGzsFz = thr_mma_gz.partition_B(esFz);
    auto tGxsXi = thr_mma_gz.partition_A(esXi);
    auto tGzgGz = thr_mma_gz.partition_C(egGz);

    auto tGxrGx = thr_mma_gx.make_fragment_C(tGxgGx);
    auto tGzrGz = thr_mma_gz.make_fragment_C(tGzgGz);
    auto grad_x = make_fragment_like(tGxrGx(_, _, _, 0, 0, 0));
    auto grad_z = make_fragment_like(tGzrGz(_, _, _, 0, 0, 0));

    clear(tGxrGx);
    clear(tGzrGz);

    for (int tile_z = 0; tile_z < NTilez; ++tile_z) {
        for (int tile_x = 0; tile_x < NTilex; ++tile_x) {
            clear(grad_x);
            clear(grad_z);

            for (int tile_k = 0; tile_k < NTilek; ++tile_k) {
                int k_tile_next =
                    ((tile_k + 1) % NTilek);  // Next k tile index in a circular manner
                int x_tile_next =
                    ((tile_k == NTilek - 1))
                        ? ((tile_x + 1) % NTilex)
                        : tile_x;  // Get next x tile index if we're at the last k tile
                int z_tile_next =
                    ((tile_k == NTilek - 1) && (tile_x == NTilex - 1))
                        ? ((tile_z + 1) % NTilez)
                        : tile_z;  // Get next z tile index if we're at the last k and x tile

                __syncthreads();
                copy(tFxrFx, tFxsFx);
                copy(tFzrFz, tFzsFz);
                __syncthreads();

                // Get next tile
                copy(tiled_copy, tFxgFx(_, _, _, z_tile_next, k_tile_next), tFxrFx);
                copy(tiled_copy, tFzgFz(_, _, _, k_tile_next, x_tile_next), tFzrFz);

                // Compute gradient for the current tile
                gemm(mmaGx, tGxsFx, tGxsGamma(_, _, _, tile_k, tile_x), grad_x);
                gemm(mmaGz, tGxsXi(_, _, _, tile_z, tile_k), tGzsFz, grad_z);
            }

            for (size_t i = 0; i < size<0>(grad_x); ++i) {
                for (size_t j = 0; j < size<1>(grad_x); ++j) {
                    for (size_t k = 0; k < size<2>(grad_x); ++k) {
                        tGxrGx(i, j, k, 0, tile_z, tile_x) +=
                            tJxgGx(i, j, k, 0, 0, tile_z, tile_x) * grad_x(i, j, k) +
                            tJxgGx(i, j, k, 0, 1, tile_x, tile_z) * grad_z(i, j, k);
                        tGxrGx(i, j, k, 1, tile_z, tile_x) +=
                            tJxgGx(i, j, k, 1, 0, tile_z, tile_x) * grad_x(i, j, k) +
                            tJxgGx(i, j, k, 1, 1, tile_x, tile_z) * grad_z(i, j, k);
                    }
                }
            }
        }
    }
    axpby(static_cast<float>(1.0), tGxrGx, static_cast<float>(0.0), tGxgGx);
}

template <typename FieldView, typename Quadrature, typename JacobianMatrixType>
class Gradient<CuteCopyFMATag, FieldView, Quadrature, JacobianMatrixType>
    : private impl::GradientBase<FieldView, Quadrature, JacobianMatrixType> {
public:
    using Base = impl::GradientBase<FieldView, Quadrature, JacobianMatrixType>;
    using typename Base::ReturnType;
    using typename Base::T;

    Gradient(const CuteCopyFMATag /*unused*/, const FieldView& field, const Quadrature& lprime,
             const JacobianMatrixType& J)
        : Base(field, lprime, J) {
    }

    static std::string name() {
        return "CuteCopyFMATag";
    }

    using Base::Base;

    ReturnType operator()() const {
        // CUTE-based implementation placeholder
        const size_t n_elements = this->n_elements_;
        const size_t nx = this->nx_;
        const size_t nz = this->nz_;

        using VecType = uint128_t;
        using TA = float;
        using TB = float;
        using TC = float;
        using CopyOp = cute::UniversalCopy<VecType>;

        const int vec_length = sizeof(VecType) / sizeof(TA);

        const auto f = this->field_.template cute_tensor<Base::ngll_, Base::ncomponents_>();
        const auto [xi, gamma] = this->lprime_.template cute_tensor<Base::ngll_>();
        const auto j = this->J_.template cute_tensor<Base::ngll_>();
        auto g = make_tensor(
            this->gradient_.data(),
            make_layout(make_shape(n_elements, Int<Base::ngll_>{}, Int<Base::ngll_>{},
                                   Int<Base::ncomponents_>{}, Int<2>{}),
                        make_shape(Int<1>{}, this->gradient_.stride_1(), this->gradient_.stride_2(),
                                   this->gradient_.stride_3(), this->gradient_.stride_4())));

        constexpr size_t bN = 4 * vec_length;
        constexpr size_t bNx = 4;
        constexpr size_t bNz = 4;
        constexpr size_t bNl = 4;

        using tile_x = Int<bN * bNx>;
        using tile_z = Int<bNz>;
        using tile_l = Int<bNl>;

        static_assert(bNx == bNz, "For symmetry, we choose bNx == bNz");

        auto cta_tiler = make_shape(Int<bN>{}, Int<bNz>{}, Int<bNx>{}, Int<bNl>{},
                                    Int<Base::ncomponents_>{}, Int<2>{});

        auto fx_smem_layout = make_layout(
            make_shape(Int<bN>{}, Int<bNz>{}, Int<bNl>{}, Int<Base::ncomponents_>{}), LayoutLeft{});
        auto fz_smem_layout = make_layout(
            make_shape(Int<bN>{}, Int<bNl>{}, Int<bNx>{}, Int<Base::ncomponents_>{}), LayoutLeft{});

        auto Xi_smem_layout = make_layout(make_shape(Int<Base::ngll_>{}, Int<Base::ngll_>{}),
                                          LayoutLeft{});  // (bNx, bNl) for tiled copy
        auto Gamma_smem_layout = make_layout(make_shape(Int<Base::ngll_>{}, Int<Base::ngll_>{}),
                                             LayoutLeft{});  // (bNz, bNl) for tiled copy

        using CopyAtom = Copy_Atom<CopyOp, TA>;

        auto tiled_copy =
            make_tiled_copy(CopyAtom{}, Layout<Shape<Int<bN / vec_length>, Int<bNz * bNl>>>{},
                            Layout<Shape<Int<vec_length>, _1>>{});

        using MMAOp = UniversalFMA<TA, TB, TC>;

        TiledMMA mmaGx = make_tiled_mma(MMAOp{}, Layout<Shape<Int<bN * bNz / 2>, Int<bNx / 2>>>{},
                                        Tile<tile_x, tile_z, tile_l>{});
        TiledMMA mmaGz = make_tiled_mma(
            MMAOp{},
            Layout<Shape<Int<bNz / 2>, Int<bN * bNx / 2>>, Stride<Int<bN * bNx / 2>, _1>>{},
            Tile<tile_z, tile_x, tile_l>{});

        static_assert(size(tiled_copy) == size(mmaGx), "Copy tile size should match MMA tile size");
        static_assert(size(tiled_copy) == size(mmaGz), "Copy tile size should match MMA tile size");
        dim3 grid((n_elements + bN - 1) / bN);
        dim3 block(static_cast<int>(bN), static_cast<int>(bNz / 2), static_cast<int>(bNx / 2));

        compute_gradient_cute_kernel_w_tiled_copy<<<grid, block>>>(
            cta_tiler, f, fx_smem_layout, fz_smem_layout, xi, gamma, Xi_smem_layout,
            Gamma_smem_layout, j, g, tiled_copy, mmaGx, mmaGz);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }

        Kokkos::fence();

        return this->gradient_;
    }

private:
    // Private helper methods for CUTE implementation can be added here
};
}  // namespace sfpp_playground
