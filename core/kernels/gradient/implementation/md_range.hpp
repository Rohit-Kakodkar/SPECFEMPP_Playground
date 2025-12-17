#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {
class MDRangeTag {};

template <typename FieldView, typename Quadrature, typename JacobianMatrixType>
class Gradient<MDRangeTag, FieldView, Quadrature, JacobianMatrixType>
    : private impl::GradientBase<FieldView, Quadrature, JacobianMatrixType> {
public:
    using Base = impl::GradientBase<FieldView, Quadrature, JacobianMatrixType>;
    using typename Base::ReturnType;
    using typename Base::T;

    Gradient(const MDRangeTag /*unused*/, const FieldView& field, const Quadrature& lprime,
             const JacobianMatrixType& J)
        : Base(field, lprime, J) {
    }

    using Base::Base;

    ReturnType operator()() const {
        // MD range policy implementation

        Kokkos::parallel_for(
            "GradientComputation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                                   {this->n_elements_, this->nz_, this->nx_}),
            KOKKOS_CLASS_LAMBDA(const size_t e, const size_t iz, const size_t ix) {
                T du_dxi[this->ncomponents_] = {static_cast<T>(0)};
                T du_dgamma[this->ncomponents_] = {static_cast<T>(0)};
                for (size_t k = 0; k < this->ngll_; ++k) {
                    for (size_t c = 0; c < this->ncomponents_; ++c) {
                        du_dxi[c] += this->lprime_.xi(ix, k) * this->field_(e, iz, k, c);
                    }
                }
                for (size_t k = 0; k < this->ngll_; ++k) {
                    for (size_t c = 0; c < this->ncomponents_; ++c) {
                        du_dgamma[c] += this->lprime_.gamma(iz, k) * this->field_(e, k, ix, c);
                    }
                }

                for (size_t c = 0; c < this->ncomponents_; ++c) {
                    this->gradient_(e, iz, ix, c, 0) = this->J_(e, iz, ix, 0, 0) * du_dxi[c] +
                                                       this->J_(e, iz, ix, 0, 1) * du_dgamma[c];
                    this->gradient_(e, iz, ix, c, 1) = this->J_(e, iz, ix, 1, 0) * du_dxi[c] +
                                                       this->J_(e, iz, ix, 1, 1) * du_dgamma[c];
                }
            });

        return this->gradient_;
    }
};
}  // namespace sfpp_playground