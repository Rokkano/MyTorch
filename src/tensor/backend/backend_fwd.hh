#pragma once

#if BACKEND_EIGEN_AVAILABLE
    template <typename T>
    struct EigenBackend;
#endif

#if BACKEND_CUDA_AVAILABLE
    template <typename T>
    struct CudaBackend;
#endif

#if BACKEND_CPP_AVAILABLE
    template <typename T>
    struct CppBackend;
    
    template <typename T>
    using DefaultBackend = CppBackend<T>;
#else
    #error Backend CPP should be available at all time.
#endif