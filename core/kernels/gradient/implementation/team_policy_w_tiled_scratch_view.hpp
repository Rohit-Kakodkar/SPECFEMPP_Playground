#pragma once

namespace sfpp_playground {

struct TeamPolicyWTiledScratchVTag {};

template <typename FieldView, typename Quadrature, typename JacobianMatrixType>
class Gradient<TeamPolicyWTiledScratchVTag, FieldView, Quadrature, JacobianMatrixType>
    : private impl::GradientBase<FieldView, Quadrature, JacobianMatrixType> {
public:
    using Base = impl::GradientBase<FieldView, Quadrature, JacobianMatrixType>;
    using typename Base::ReturnType;
    using typename Base::T;

    Gradient(const TeamPolicyWTiledScratchVTag /*unused*/, const FieldView& field,
             const Quadrature& lprime, const JacobianMatrixType& J)
        : Base(field, lprime, J) {
    }

    static std::string name() {
        return "TeamPolicyWTiledScratchVTag";
    }

    using Base::Base;

    ReturnType operator()() const {
        // Team policy with tiled scratch memory implementation
#if defined(KOKKOS_ENABLE_CUDA)
        constexpr static size_t tile_size[] = {32, 8, 8};  // {elements, z, x}
#else
        constexpr static size_t tile_size[] = {1, 8, 8};  // {elements, z, x}
#endif
        constexpr static size_t tile_size_e = tile_size[0];
        constexpr static size_t tile_size_z = tile_size[1];
        constexpr static size_t tile_size_x = tile_size[2];
        const size_t nteams =
            this->n_elements_ / tile_size[0] + (this->n_elements_ % tile_size[0] != 0 ? 1 : 0);

        using execution_space = typename Base::execution_space;

        using ScratchSpaceType = Kokkos::View<T[tile_size[0]][tile_size[1]][tile_size[2]][1],
                                              typename execution_space::scratch_memory_space,
                                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using LagrangeViewType = Kokkos::View<T[tile_size[2]][tile_size[1]],
                                              typename execution_space::scratch_memory_space,
                                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        const size_t scratch_size =
            2 * ScratchSpaceType::shmem_size() +
            2 * LagrangeViewType::shmem_size();  // For field slices & lprime

#if defined(KOKKOS_ENABLE_CUDA)
        const auto policy = Kokkos::TeamPolicy<>(nteams, Kokkos::AUTO, Kokkos::AUTO)
                                .set_scratch_size(0, Kokkos::PerTeam(scratch_size));
#else
        const auto policy = Kokkos::TeamPolicy<>(nteams, Kokkos::AUTO, Kokkos::AUTO)
                                .set_scratch_size(0, Kokkos::PerTeam(scratch_size));
#endif

        Kokkos::parallel_for(
            "GradientComputationTeamPolicyWTiledScratch", policy,
            KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
                const size_t team_id = team.league_rank();
                const size_t start_e = team_id * tile_size_e;
                const size_t end_e = (start_e + tile_size_e < this->n_elements_)
                                         ? (start_e + tile_size_e)
                                         : this->n_elements_;
                const size_t local_chunk_size = end_e - start_e;

                ScratchSpaceType field_x_scratch(team.team_scratch(0));
                ScratchSpaceType field_z_scratch(team.team_scratch(0));
                LagrangeViewType xi_(team.team_scratch(0));
                LagrangeViewType gamma_(team.team_scratch(0));

                for (size_t tile_k = 0; tile_k < this->nx_; tile_k += tile_size_x) {
                    const size_t tile_k_start = tile_k;
                    const size_t tile_k_end = (tile_k_start + tile_size_x < this->nx_)
                                                  ? (tile_k_start + tile_size_x)
                                                  : this->nx_;
                    for (size_t tile_z = 0; tile_z < this->nz_; tile_z += tile_size_z) {
                        const size_t tile_iz_start = tile_z;
                        const size_t tile_iz_end = (tile_iz_start + tile_size_z < this->nz_)
                                                       ? (tile_iz_start + tile_size_z)
                                                       : this->nz_;

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team, tile_size_e * tile_size_z),
                            [&](const size_t idx) {
                                const size_t local_e = idx % tile_size_e;
                                const size_t iz = idx / tile_size_e;

                                if (local_e > local_chunk_size - 1 ||
                                    iz > tile_iz_end - tile_iz_start - 1)
                                    return;

                                for (size_t c = 0; c < this->ncomponents_; ++c) {
                                    for (size_t k = 0; k < tile_size_x; ++k) {
                                        field_x_scratch(local_e, iz, k, c) =
                                            this->field_(start_e + local_e, iz + tile_iz_start,
                                                         k + tile_k_start, c);
                                        gamma_(iz, k) = this->lprime_.gamma(iz + tile_iz_start,
                                                                            k + tile_k_start);
                                    }
                                }
                            });

                        for (size_t tile_x = 0; tile_x < this->nx_; tile_x += tile_size_x) {
                            const size_t tile_ix_start = tile_x;
                            const size_t tile_ix_end = (tile_ix_start + tile_size_x < this->nx_)
                                                           ? (tile_ix_start + tile_size_x)
                                                           : this->nx_;

                            Kokkos::parallel_for(
                                Kokkos::TeamThreadRange(team, tile_size_e * tile_size_x),
                                [&](const size_t idx) {
                                    const size_t local_e = idx % tile_size_e;
                                    const size_t ix = idx / tile_size_e;

                                    if (local_e > local_chunk_size - 1 ||
                                        ix > tile_ix_end - tile_ix_start - 1)
                                        return;

                                    for (size_t c = 0; c < this->ncomponents_; ++c) {
                                        for (size_t k = 0; k < tile_size_z; ++k) {
                                            field_z_scratch(local_e, k, ix, c) =
                                                this->field_(start_e + local_e, k + tile_iz_start,
                                                             ix + tile_ix_start, c);
                                            xi_(ix, k) = this->lprime_.xi(ix + tile_ix_start,
                                                                          k + tile_k_start);
                                        }
                                    }
                                });

                            team.team_barrier();

                            Kokkos::parallel_for(
                                Kokkos::TeamThreadRange(team,
                                                        tile_size_e * tile_size_z * tile_size_x),
                                [&](const size_t idx) {
                                    const size_t local_e = idx % tile_size_e;
                                    const size_t iz = (idx / tile_size_e) % tile_size_z;
                                    const size_t ix = idx / (tile_size_e * tile_size_z);

                                    if (local_e > local_chunk_size - 1 ||
                                        iz > tile_iz_end - tile_iz_start - 1 ||
                                        ix > tile_ix_end - tile_ix_start - 1)
                                        return;

                                    T du_dxi[this->ncomponents_] = {static_cast<T>(0)};
                                    T du_dgamma[this->ncomponents_] = {static_cast<T>(0)};
                                    for (size_t k = 0; k < tile_size_x; ++k) {
                                        for (size_t c = 0; c < this->ncomponents_; ++c) {
                                            du_dxi[c] +=
                                                xi_(ix, k) * field_x_scratch(local_e, iz, k, c);
                                        }
                                    }
                                    for (size_t k = 0; k < tile_size_z; ++k) {
                                        for (size_t c = 0; c < this->ncomponents_; ++c) {
                                            du_dgamma[c] +=
                                                gamma_(iz, k) * field_z_scratch(local_e, k, ix, c);
                                        }
                                    }

                                    for (size_t c = 0; c < this->ncomponents_; ++c) {
                                        this->gradient_(start_e + local_e, iz + tile_iz_start,
                                                        ix + tile_ix_start, c, 0) +=
                                            this->J_(start_e + local_e, iz + tile_iz_start,
                                                     ix + tile_ix_start, 0, 0) *
                                                du_dxi[c] +
                                            this->J_(start_e + local_e, iz + tile_iz_start,
                                                     ix + tile_ix_start, 0, 1) *
                                                du_dgamma[c];
                                        this->gradient_(start_e + local_e, iz + tile_iz_start,
                                                        ix + tile_ix_start, c, 1) +=
                                            this->J_(start_e + local_e, iz + tile_iz_start,
                                                     ix + tile_ix_start, 1, 0) *
                                                du_dxi[c] +
                                            this->J_(start_e + local_e, iz + tile_iz_start,
                                                     ix + tile_ix_start, 1, 1) *
                                                du_dgamma[c];
                                    }
                                });
                        }
                    }
                }
            });

        Kokkos::fence();

        return this->gradient_;
    }
};

}  // namespace sfpp_playground
