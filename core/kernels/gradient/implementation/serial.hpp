#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {
class SerialTag {};

template <typename FieldView, typename Quadrature, typename JacobianMatrixType>
class Gradient<SerialTag, FieldView, Quadrature, JacobianMatrixType>
    : private impl::GradientBase<FieldView, Quadrature, JacobianMatrixType> {
public:
    using Base = impl::GradientBase<FieldView, Quadrature, JacobianMatrixType>;
    using typename Base::ReturnType;
    using typename Base::T;

    Gradient(const SerialTag /*unused*/, const FieldView& field, const Quadrature& lprime,
             const JacobianMatrixType& J)
        : Base(field, lprime, J) {
    }

    using Base::Base;

    ReturnType operator()() const {
        // MD range policy implementation

        for (size_t e = 0; e < this->n_elements_; ++e) {
            for (size_t iz = 0; iz < this->nz_; ++iz) {
                for (size_t ix = 0; ix < this->nx_; ++ix) {
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
                }
            }
        }

        return this->gradient_;
    }
};

}  // namespace sfpp_playground