#pragma once

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <memory>

class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        context_ = std::make_unique<Kokkos::ScopeGuard>();
    }
    void TearDown() override {
        context_.reset();
    }

private:
    static std::unique_ptr<Kokkos::ScopeGuard> context_;
};

std::unique_ptr<Kokkos::ScopeGuard> KokkosEnvironment::context_ = nullptr;
