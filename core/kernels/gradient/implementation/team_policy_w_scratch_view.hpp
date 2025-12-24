#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {
struct TeamPolicyWScratchVTag {};

template <typename FieldView, typename Quadrature, typename JacobianMatrixType>
class Gradient<TeamPolicyWScratchVTag, FieldView, Quadrature, JacobianMatrixType>
    : private impl::GradientBase<FieldView, Quadrature, JacobianMatrixType> {
public:
    using Base = impl::GradientBase<FieldView, Quadrature, JacobianMatrixType>;
    using typename Base::ReturnType;
    using typename Base::T;

    Gradient(const TeamPolicyWScratchVTag /*unused*/, const FieldView& field,
             const Quadrature& lprime, const JacobianMatrixType& J)
        : Base(field, lprime, J) {
    }

    static std::string name() {
        return "TeamPolicyWScratchVTag";
    }

    using Base::Base;

    ReturnType operator()() const {
        // Team policy with scratch memory implementation

        using execution_space = typename Base::execution_space;

        using ScratchSpaceType = Kokkos::View<T***, typename execution_space::scratch_memory_space,
                                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        const size_t scratch_size = ScratchSpaceType::shmem_size(
            this->ngll_, this->ngll_, this->ncomponents_);  // For storing the field slice

        Kokkos::parallel_for(
            "GradientComputationTeamPolicyWScratch",
            Kokkos::TeamPolicy<>(this->n_elements_, Kokkos::AUTO, Kokkos::AUTO)
                .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
                const size_t e = team.league_rank();

                ScratchSpaceType field_scratch(team.team_scratch(0), this->ngll_, this->ngll_,
                                               this->ncomponents_);
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, this->nz_ * this->nx_),
                                     [&](const size_t idx) {
                                         const size_t iz = idx % this->nz_;
                                         const size_t ix = idx / this->nz_;
                                         for (size_t c = 0; c < this->ncomponents_; ++c) {
                                             field_scratch(iz, ix, c) = this->field_(e, iz, ix, c);
                                         }
                                     });

                team.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, this->nz_ * this->nx_), [&](const size_t idx) {
                        const size_t iz = idx % this->nz_;
                        const size_t ix = idx / this->nz_;

                        T du_dxi[this->ncomponents_] = {static_cast<T>(0)};
                        T du_dgamma[this->ncomponents_] = {static_cast<T>(0)};
                        for (size_t k = 0; k < this->ngll_; ++k) {
                            for (size_t c = 0; c < this->ncomponents_; ++c) {
                                du_dxi[c] += this->lprime_.xi(ix, k) * field_scratch(iz, k, c);
                            }
                        }
                        for (size_t k = 0; k < this->ngll_; ++k) {
                            for (size_t c = 0; c < this->ncomponents_; ++c) {
                                du_dgamma[c] +=
                                    this->lprime_.gamma(iz, k) * field_scratch(k, ix, c);
                            }
                        }

                        for (size_t c = 0; c < this->ncomponents_; ++c) {
                            this->gradient_(e, iz, ix, c, 0) =
                                this->J_(e, iz, ix, 0, 0) * du_dxi[c] +
                                this->J_(e, iz, ix, 0, 1) * du_dgamma[c];
                            this->gradient_(e, iz, ix, c, 1) =
                                this->J_(e, iz, ix, 1, 0) * du_dxi[c] +
                                this->J_(e, iz, ix, 1, 1) * du_dgamma[c];
                        }
                    });
            });

        Kokkos::fence();

        return this->gradient_;
    }
};

}  // namespace sfpp_playground
