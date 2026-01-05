#pragma once

#include <Kokkos_Core.hpp>

namespace sfpp_playground {
struct SerialTag {};

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

    static std::string name() {
        return "SerialTag";
    }

    using Base::Base;

    ReturnType operator()() const {
        // MD range policy implementation

        const auto field_host = create_mirror_view_and_copy(Kokkos::HostSpace{}, this->field_);
        const auto lprime_host = create_mirror_view_and_copy(Kokkos::HostSpace{}, this->lprime_);
        const auto J_host = create_mirror_view_and_copy(Kokkos::HostSpace{}, this->J_);
        const auto gradient_host =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, this->gradient_);

        for (size_t e = 0; e < this->n_elements_; ++e) {
            for (size_t iz = 0; iz < this->nz_; ++iz) {
                for (size_t ix = 0; ix < this->nx_; ++ix) {
                    T du_dxi[this->ncomponents_] = {static_cast<T>(0)};
                    T du_dgamma[this->ncomponents_] = {static_cast<T>(0)};
                    for (size_t k = 0; k < this->ngll_; ++k) {
                        for (size_t c = 0; c < this->ncomponents_; ++c) {
                            du_dxi[c] += lprime_host.xi(iz, k) * field_host(e, k, ix, c);
                        }
                    }
                    for (size_t k = 0; k < this->ngll_; ++k) {
                        for (size_t c = 0; c < this->ncomponents_; ++c) {
                            du_dgamma[c] += lprime_host.gamma(iz, k) * field_host(e, k, ix, c);
                        }
                    }

                    for (size_t c = 0; c < this->ncomponents_; ++c) {
                        gradient_host(e, iz, ix, c, 0) = J_host(e, iz, ix, 0, 0) * du_dxi[c] +
                                                         J_host(e, iz, ix, 0, 1) * du_dgamma[c];
                        gradient_host(e, iz, ix, c, 1) = J_host(e, iz, ix, 1, 0) * du_dxi[c] +
                                                         J_host(e, iz, ix, 1, 1) * du_dgamma[c];
                    }
                }
            }
        }

        Kokkos::deep_copy(this->gradient_, gradient_host);

        return this->gradient_;
    }
};

}  // namespace sfpp_playground
