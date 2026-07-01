#pragma once

#if BACKEND_EIGEN_AVAILABLE
#include "src/tensor/backend/eigen/backend.hh"
#include "src/tensor/backend/eigen/linear.hxx"
#endif

#if BACKEND_CUDA_AVAILABLE
// ...
#endif

#if BACKEND_CPP_AVAILABLE
#include "src/tensor/backend/cpp/backend.hh"
#include "src/tensor/backend/cpp/linear.hxx"
#include "src/tensor/backend/cpp/math.hxx"
#endif