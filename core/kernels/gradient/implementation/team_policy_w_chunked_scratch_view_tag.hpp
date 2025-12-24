#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {
struct TeamPolicyWChunkedScratchVTag {};

template <typename FieldView, typename Quadrature, typename JacobianMatrixType>
class Gradient<TeamPolicyWChunkedScratchVTag, FieldView, Quadrature, JacobianMatrixType>
    : private impl::GradientBase<FieldView, Quadrature, JacobianMatrixType> {
public:
    using Base = impl::GradientBase<FieldView, Quadrature, JacobianMatrixType>;
    using typename Base::ReturnType;
    using typename Base::T;

    Gradient(const TeamPolicyWChunkedScratchVTag /*unused*/, const FieldView& field,
             const Quadrature& lprime, const JacobianMatrixType& J)
        : Base(field, lprime, J) {
    }

    static std::string name() {
        return "TeamPolicyWChunkedScratchVTag";
    }

    using Base::Base;

    ReturnType operator()() const {
        // Team policy with chunked scratch memory implementation
        constexpr static size_t chunk_size = 32;
        const size_t nteams =
            this->n_elements_ / chunk_size + (this->n_elements_ % chunk_size != 0 ? 1 : 0);

        using execution_space = typename Base::execution_space;

        using ScratchSpaceType = Kokkos::View<T****, typename execution_space::scratch_memory_space,
                                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        const size_t scratch_size = ScratchSpaceType::shmem_size(
            chunk_size, this->ngll_, this->ngll_, this->ncomponents_);  // For storing field slices

        Kokkos::parallel_for(
            "GradientComputationTeamPolicyWChunkedScratch",
            Kokkos::TeamPolicy<>(nteams, Kokkos::AUTO, Kokkos::AUTO)
                .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
                const size_t team_id = team.league_rank();
                const size_t start_e = team_id * chunk_size;
                const size_t end_e = (start_e + chunk_size < this->n_elements_)
                                         ? (start_e + chunk_size)
                                         : this->n_elements_;
                const size_t local_chunk_size = end_e - start_e;
                ScratchSpaceType field_scratch(team.team_scratch(0), local_chunk_size, this->ngll_,
                                               this->ngll_, this->ncomponents_);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, local_chunk_size * this->nz_ * this->nx_),
                    [&](const size_t idx) {
                        const size_t local_e = idx % local_chunk_size;
                        const size_t iz = (idx / local_chunk_size) % this->nz_;
                        const size_t ix = idx / (local_chunk_size * this->nz_);
                        for (size_t c = 0; c < this->ncomponents_; ++c) {
                            field_scratch(local_e, iz, ix, c) =
                                this->field_(start_e + local_e, iz, ix, c);
                        }
                    });

                team.team_barrier();
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, local_chunk_size * this->nz_ * this->nx_),
                    [&](const size_t idx) {
                        const size_t local_e = idx % local_chunk_size;
                        const size_t iz = (idx / local_chunk_size) % this->nz_;
                        const size_t ix = idx / (local_chunk_size * this->nz_);
                        T du_dxi[this->ncomponents_] = {static_cast<T>(0)};
                        T du_dgamma[this->ncomponents_] = {static_cast<T>(0)};
                        for (size_t k = 0; k < this->ngll_; ++k) {
                            for (size_t c = 0; c < this->ncomponents_; ++c) {
                                du_dxi[c] +=
                                    this->lprime_.xi(ix, k) * field_scratch(local_e, iz, k, c);
                            }
                        }
                        for (size_t k = 0; k < this->ngll_; ++k) {
                            for (size_t c = 0; c < this->ncomponents_; ++c) {
                                du_dgamma[c] +=
                                    this->lprime_.gamma(iz, k) * field_scratch(local_e, k, ix, c);
                            }
                        }

                        for (size_t c = 0; c < this->ncomponents_; ++c) {
                            this->gradient_(start_e + local_e, iz, ix, c, 0) =
                                this->J_(start_e + local_e, iz, ix, 0, 0) * du_dxi[c] +
                                this->J_(start_e + local_e, iz, ix, 0, 1) * du_dgamma[c];
                            this->gradient_(start_e + local_e, iz, ix, c, 1) =
                                this->J_(start_e + local_e, iz, ix, 1, 0) * du_dxi[c] +
                                this->J_(start_e + local_e, iz, ix, 1, 1) * du_dgamma[c];
                        }
                    });
            });

        Kokkos::fence();

        return this->gradient_;
    }
};

}  // namespace sfpp_playground
