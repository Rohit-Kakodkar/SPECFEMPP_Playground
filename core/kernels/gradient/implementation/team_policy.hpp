#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {
struct TeamPolicyTag {
    static std::string name() {
        return "TeamPolicyTag";
    }
};

template <typename FieldView, typename Quadrature, typename JacobianMatrixType>
class Gradient<TeamPolicyTag, FieldView, Quadrature, JacobianMatrixType>
    : private impl::GradientBase<FieldView, Quadrature, JacobianMatrixType> {
public:
    using Base = impl::GradientBase<FieldView, Quadrature, JacobianMatrixType>;
    using typename Base::ReturnType;
    using typename Base::T;

    Gradient(const TeamPolicyTag /*unused*/, const FieldView& field, const Quadrature& lprime,
             const JacobianMatrixType& J)
        : Base(field, lprime, J) {
    }

    using Base::Base;

    ReturnType operator()() const {
        // Team policy implementation

        Kokkos::parallel_for(
            "GradientComputationTeamPolicy",
            Kokkos::TeamPolicy<>(this->n_elements_, Kokkos::AUTO, Kokkos::AUTO),
            KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
                const size_t e = team.league_rank();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, this->nx_ * this->nz_), [&](const size_t idx) {
                        const size_t iz = idx % this->nz_;
                        const size_t ix = idx / this->nz_;

                        T du_dxi[this->ncomponents_] = {static_cast<T>(0)};
                        T du_dgamma[this->ncomponents_] = {static_cast<T>(0)};
                        for (size_t k = 0; k < this->ngll_; ++k) {
                            for (size_t c = 0; c < this->ncomponents_; ++c) {
                                du_dxi[c] += this->lprime_.xi(ix, k) * this->field_(e, iz, k, c);
                            }
                        }
                        for (size_t k = 0; k < this->ngll_; ++k) {
                            for (size_t c = 0; c < this->ncomponents_; ++c) {
                                du_dgamma[c] +=
                                    this->lprime_.gamma(iz, k) * this->field_(e, k, ix, c);
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

        return this->gradient_;
    }
};

}  // namespace sfpp_playground
