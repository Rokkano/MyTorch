#pragma once

#include "src/tensor/backend/backend.hh"
#include "src/tensor/tensor_fwd.hh"

#include <optional>

template <typename T>
struct EigenBackend
{
    using TStorage = std::vector<T>;

    static TStorage allocate(std::size_t n) { return TStorage(n); }

    static T *data_ptr(TStorage &s) { return s.data(); }

    static std::vector<T>::iterator begin(TStorage &s) { return s.begin(); }
    static std::vector<T>::iterator const_begin(TStorage &s) { return s.const_begin(); }
    static std::vector<T>::iterator end(TStorage &s) { return s.end(); }
    static std::vector<T>::iterator const_end(TStorage &s) { return s.const_end(); }
};

#include "eigen_linear.hxx"