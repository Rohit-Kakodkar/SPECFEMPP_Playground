#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {
struct RangeTag {};

template <typename FieldView, typename Quadrature, typename JacobianMatrixType>
class Gradient<RangeTag, FieldView, Quadrature, JacobianMatrixType>
    : private impl::GradientBase<FieldView, Quadrature, JacobianMatrixType> {
public:
    using Base = impl::GradientBase<FieldView, Quadrature, JacobianMatrixType>;
    using typename Base::ReturnType;
    using typename Base::T;

    Gradient(const RangeTag /*unused*/, const FieldView& field, const Quadrature& lprime,
             const JacobianMatrixType& J)
        : Base(field, lprime, J) {
    }

    static std::string name() {
        return "RangeTag";
    }

    using Base::Base;

    ReturnType operator()() const {
        // Range policy implementation - flatten 3D iteration space

        const size_t total_iterations = this->n_elements_ * this->nz_ * this->nx_;

        Kokkos::parallel_for(
            "GradientComputation", Kokkos::RangePolicy<>(0, total_iterations),
            KOKKOS_CLASS_LAMBDA(const size_t idx) {
                // Decompose flat index into (e, iz, ix)
                const size_t e = idx % this->n_elements_;
                const size_t iz = (idx / this->n_elements_) % this->nz_;
                const size_t ix = idx / (this->n_elements_ * this->nz_);

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

        Kokkos::fence();

        return this->gradient_;
    }
};
}  // namespace sfpp_playground
