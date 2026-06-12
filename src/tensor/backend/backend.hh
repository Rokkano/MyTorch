#pragma once

#if BACKEND_EIGEN
    #include "src/tensor/backend/eigen/backend.hh"
    #include "src/tensor/backend/eigen/linear.hxx"
#elif BACKEND_CPP
    #include "src/tensor/backend/cpp/backend.hh"
    #include "src/tensor/backend/cpp/linear.hxx"
    #include "src/tensor/backend/cpp/math.hxx"
#else
    #error No backend available
#endif