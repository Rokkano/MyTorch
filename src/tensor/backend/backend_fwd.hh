#pragma once

#if BACKEND_EIGEN
    template <typename T>
    struct EigenBackend;
#elif BACKEND_CPP
    template <typename T>
    struct CppBackend;
#else
    #error No backend available
#endif