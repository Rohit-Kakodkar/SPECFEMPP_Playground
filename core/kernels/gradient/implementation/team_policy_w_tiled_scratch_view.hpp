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
        constexpr static size_t tile_size[] = {64, 2, 2};  // {elements, z, x}
        constexpr static size_t tile_size_z = tile_size[1];
        constexpr static size_t tile_size_x = tile_size[2];
        const size_t nteams =
            this->n_elements_ / tile_size[0] + (this->n_elements_ % tile_size[0] != 0 ? 1 : 0);

        const size_t ntiles_z = this->nz_ / tile_size[1] + (this->nz_ % tile_size[1] != 0 ? 1 : 0);
        const size_t ntiles_x = this->nx_ / tile_size[2] + (this->nx_ % tile_size[2] != 0 ? 1 : 0);

        using execution_space = typename Base::execution_space;

        using ScratchSpaceType = Kokkos::View<T[tile_size[0]][tile_size[1]][tile_size[2]][1],
                                              typename execution_space::scratch_memory_space,
                                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        const size_t scratch_size = ScratchSpaceType::shmem_size();  // For storing field tiles

        Kokkos::parallel_for(
            "GradientComputationTeamPolicyWTiledScratch",
            Kokkos::TeamPolicy<>(nteams, Kokkos::AUTO, Kokkos::AUTO)
                .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
                const size_t team_id = team.league_rank();
                const size_t start_e = team_id * tile_size[0];
                const size_t end_e = (start_e + tile_size[0] < this->n_elements_)
                                         ? (start_e + tile_size[0])
                                         : this->n_elements_;
                const size_t local_chunk_size = end_e - start_e;

                ScratchSpaceType field_scratch(team.team_scratch(0));

                for (size_t tile_z = 0; tile_z < this->nz_; tile_z += tile_size_z) {
                    const size_t tile_iz_start = tile_z;
                    const size_t tile_iz_end = (tile_iz_start + tile_size_z < this->nz_)
                                                   ? (tile_iz_start + tile_size_z)
                                                   : this->nz_;
                    for (size_t tile_x = 0; tile_x < this->nx_; tile_x += tile_size_x) {
                        const size_t tile_ix_start = tile_x;
                        const size_t tile_ix_end = (tile_ix_start + tile_size_x < this->nx_)
                                                       ? (tile_ix_start + tile_size_x)
                                                       : this->nx_;
                        const auto field_subview = Kokkos::subview(
                            this->field_, Kokkos::make_pair(start_e, end_e),
                            Kokkos::make_pair(tile_iz_start, tile_iz_end),
                            Kokkos::make_pair(tile_ix_start, tile_ix_end), Kokkos::ALL());
                        const auto xi_subview = Kokkos::subview(
                            this->lprime_.xi(), Kokkos::pair(tile_ix_start, tile_ix_end),
                            Kokkos::pair(tile_iz_start, tile_iz_end));
                        const auto gamma_subview = Kokkos::subview(
                            this->lprime_.gamma(), Kokkos::pair(tile_iz_start, tile_iz_end),
                            Kokkos::pair(tile_ix_start, tile_ix_end));

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team, local_chunk_size *
                                                              (tile_iz_end - tile_iz_start) *
                                                              (tile_ix_end - tile_ix_start)),
                            [&](const size_t idx) {
                                const size_t local_e = idx % local_chunk_size;
                                const size_t iz =
                                    (idx / local_chunk_size) % (tile_iz_end - tile_iz_start);
                                const size_t ix =
                                    idx / (local_chunk_size * (tile_iz_end - tile_iz_start));
                                for (size_t c = 0; c < this->ncomponents_; ++c) {
                                    field_scratch(local_e, iz, ix, c) =
                                        field_subview(local_e, iz, ix, c);
                                }
                            });

                        team.team_barrier();

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team, local_chunk_size *
                                                              (tile_iz_end - tile_iz_start) *
                                                              (tile_ix_end - tile_ix_start)),
                            [&](const size_t idx) {
                                const size_t local_e = idx % local_chunk_size;
                                const size_t iz =
                                    (idx / local_chunk_size) % (tile_iz_end - tile_iz_start);
                                const size_t ix =
                                    idx / (local_chunk_size * (tile_iz_end - tile_iz_start));
                                T du_dxi[this->ncomponents_] = {static_cast<T>(0)};
                                T du_dgamma[this->ncomponents_] = {static_cast<T>(0)};
                                for (size_t k = 0; k < 2; ++k) {
                                    for (size_t c = 0; c < this->ncomponents_; ++c) {
                                        du_dxi[c] +=
                                            xi_subview(ix, k) * field_scratch(local_e, k, ix, c);
                                    }
                                }
                                for (size_t k = 0; k < 2; ++k) {
                                    for (size_t c = 0; c < this->ncomponents_; ++c) {
                                        du_dgamma[c] +=
                                            gamma_subview(iz, k) * field_scratch(local_e, iz, k, c);
                                    }
                                }

                                for (size_t c = 0; c < this->ncomponents_; ++c) {
                                    this->gradient_(start_e + local_e, tile_iz_start + iz,
                                                    tile_ix_start + ix, c, 0) +=
                                        this->J_(start_e + local_e, tile_iz_start + iz,
                                                 tile_ix_start + ix, 0, 0) *
                                            du_dxi[c] +
                                        this->J_(start_e + local_e, tile_iz_start + iz,
                                                 tile_ix_start + ix, 0, 1) *
                                            du_dgamma[c];
                                    this->gradient_(start_e + local_e, tile_iz_start + iz,
                                                    tile_ix_start + ix, c, 1) +=
                                        this->J_(start_e + local_e, tile_iz_start + iz,
                                                 tile_ix_start + ix, 1, 0) *
                                            du_dxi[c] +
                                        this->J_(start_e + local_e, tile_iz_start + iz,
                                                 tile_ix_start + ix, 1, 1) *
                                            du_dgamma[c];
                                }
                            });
                    }
                }
            });

        Kokkos::fence();

        return this->gradient_;
    }
};

}  // namespace sfpp_playground
