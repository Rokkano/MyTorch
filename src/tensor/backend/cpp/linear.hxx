#pragma once

#include "backend.hh"
#include "src/exception/tensor.hh"

#include <cmath>
#include <format>
#include <optional>
#include <typeinfo>

template <typename T>
CppBackend<T>::TStorage CppBackend<T>::dot(const TStorage &lhsStorage, const TShape &lhsShape, const TStorage &rhsStorage, const TShape &rhsShape)
requires std::is_arithmetic_v<T>
{
    if (lhsShape.size() != 1)
        throw TensorInvalidShapeException(std::format("Dot product only applies for 1-dimensional tensors : {}.",
                                                      Tensor<T, CppBackend>::shapeToStr(lhsShape)));
    if (rhsShape.size() != 1)
        throw TensorInvalidShapeException(std::format("Dot product only applies for 1-dimensional tensors : {}.",
                                                      Tensor<T, CppBackend>::shapeToStr(rhsShape)));
    if (lhsShape[0] != rhsShape[0])
        throw TensorInvalidShapeException(std::format("Lengths are incompatible for dot product : {} and {}.",
                                                      Tensor<T, CppBackend>::shapeToStr(rhsShape),
                                                      Tensor<T, CppBackend>::shapeToStr(lhsShape)));

    CppBackend<T>::TStorage newStorage = CppBackend<T>::allocate(1);
    newStorage[0] = 0;
    for (std::size_t i = 0; i < lhsShape[0]; i++)
        newStorage[0] += lhsStorage[i] * rhsStorage[i];
    return newStorage;
}


template <typename T>
CppBackend<T>::TStorage CppBackend<T>::mvm(const TStorage &lhsStorage, const TShape &lhsShape, const TStorage &rhsStorage, const TShape &rhsShape) 
requires std::is_arithmetic_v<T>
{
    // matrix vector multiplication
    if (lhsShape.size() != 2)
        throw TensorInvalidShapeException(std::format("mvm only applies for 2-dimensional tensors for lhs : {}.",
                                                      Tensor<T, CppBackend>::shapeToStr(lhsShape)));
    if (rhsShape.size() != 1)
        throw TensorInvalidShapeException(std::format("mvm only applies for 1-dimensional tensors for rhs : {}.",
                                                      Tensor<T, CppBackend>::shapeToStr(rhsShape)));
    if (lhsShape[1] != rhsShape[0])
        throw TensorInvalidShapeException(std::format("Tensors are not compatible for matrix-vector "
                                                      "multiplication : {} and {}.",
                                                      Tensor<T, CppBackend>::shapeToStr(lhsShape),
                                                      Tensor<T, CppBackend>::shapeToStr(rhsShape)));

    CppBackend<T>::TStorage newStorage = CppBackend<T>::allocate(lhsShape[0]);
    for (std::size_t y = 0; y < lhsShape[0]; y++)
        for (std::size_t x = 0; x < lhsShape[1]; x++)
            newStorage[y] += lhsStorage[x + y * lhsShape[1]] * rhsStorage[x];
    return newStorage;
}

template <typename T>
CppBackend<T>::TStorage CppBackend<T>::mm(const TStorage &lhsStorage, const TShape &lhsShape, const TStorage &rhsStorage, const TShape &rhsShape) 
requires std::is_arithmetic_v<T>
{
    // matrix multiplication (2d tensors)
    if (lhsShape.size() != 2)
        throw TensorInvalidShapeException(std::format("mm only applies for 2-dimensional tensors : {}.",
                                                      Tensor<T, CppBackend>::shapeToStr(lhsShape)));
    if (rhsShape.size() != 2)
        throw TensorInvalidShapeException(std::format("mm only applies for 2-dimensional tensors : {}.",
                                                      Tensor<T, CppBackend>::shapeToStr(rhsShape)));
    if (lhsShape[1] != rhsShape[0])
        throw TensorInvalidShapeException(
            std::format("Tensors are not compatible for matrix multiplication : {} and {}.",
                        Tensor<T, CppBackend>::shapeToStr(lhsShape),
                        Tensor<T, CppBackend>::shapeToStr(rhsShape)));

    std::size_t numel = lhsShape[0] * rhsShape[1];
    std::vector<std::size_t> newShape = {lhsShape[0], rhsShape[1]};
    CppBackend<T>::TStorage newStorage = CppBackend<T>::allocate(numel);
    for(std::size_t i = 0; i < numel; i++)
        newStorage[i] = 0;

    for (std::size_t y = 0; y < newShape[1]; y++)
        for (std::size_t x = 0; x < newShape[0]; x++)
            for (std::size_t k = 0; k < lhsShape[1]; k++)
                newStorage[x * newShape[1] + y] +=
                    lhsStorage[x * lhsShape[1] + k] * rhsStorage[k * rhsShape[1] + y];
    return newStorage;
}

// template <typename T>
// CppBackend<T>::TStorage CppBackend<T>::omm(const TStorage &lhsStorage, const TShape &lhsShape, const TStorage &rhsStorage, const TShape &rhsShape) 
// requires std::is_arithmetic_v<T>
// {
//     // optimized matrix multiplication (2d tensors) with rhs transpose for
//     // quicker data reading
//     if (lhsShape.size() != 2)
//         throw TensorInvalidShapeException(std::format("mm only applies for 2-dimensional tensors : {}.",
//                                                       Tensor<T, CppBackend>::shapeToStr(lhsShape)));
//     if (rhsShape.size() != 2)
//         throw TensorInvalidShapeException(std::format("mm only applies for 2-dimensional tensors : {}.",
//                                                       Tensor<T, CppBackend>::shapeToStr(rhsShape)));
//     if (lhsShape[1] != rhsShape[0])
//         throw TensorInvalidShapeException(
//             std::format("Tensors are not compatible for matrix multiplication : {} and {}.",
//                         Tensor<T, CppBackend>::shapeToStr(lhsShape),
//                         Tensor<T, CppBackend>::shapeToStr(rhsShape)));

//     Tensor<T, CppBackend> tensor = Tensor<T, CppBackend>({lhsShape[0], rhsShape[1]});
//     Tensor<T, CppBackend> rhs_t = rhs.transpose();
//     for (std::size_t y = 0; y < lhsShape[0]; y++)
//         for (std::size_t x = 0; x < rhsShape[1]; x++)
//             for (std::size_t k = 0; k < lhsShape[1]; k++)
//                 tensor.buffer_[x + y * rhsShape[1]] +=
//                     lhs.buffer_[k + y * lhsShape[1]] * rhs_t.buffer_[k + x * rhs_t.shape_[1]];

//     return tensor;
// }

template <typename T>
CppBackend<T>::TStorage CppBackend<T>::bmm(const TStorage &lhsStorage, const TShape &lhsShape, const TStorage &rhsStorage, const TShape &rhsShape) 
requires std::is_arithmetic_v<T>
{
    if (lhsShape.size() < 3)
        throw TensorInvalidShapeException(std::format("bmm only applies for 3 or + dimensional tensors (batched): {}.",
                                                      Tensor<T, CppBackend>::shapeToStr(lhsShape)));
    if (rhsShape.size() < 3)
        throw TensorInvalidShapeException(std::format("bmm only applies for 3 or + dimensional tensors (batched): {}.",
                                                      Tensor<T, CppBackend>::shapeToStr(rhsShape)));
    if (lhsShape[lhsShape.size() - 1] != rhsShape[rhsShape.size() - 2])
        throw TensorInvalidShapeException(std::format("Tensors are not compatible for batched matrix "
                                                      "multiplication : {} and {}.",
                                                      Tensor<T, CppBackend>::shapeToStr(lhsShape),
                                                      Tensor<T, CppBackend>::shapeToStr(rhsShape)));
    if (lhsShape.size() != rhsShape.size())
        throw TensorInvalidShapeException(std::format("Tensors dimensions are not compatible for batched matrix "
                                                      "multiplication (use broadcasting) : {} and {}.",
                                                      Tensor<T, CppBackend>::shapeToStr(lhsShape),
                                                      Tensor<T, CppBackend>::shapeToStr(rhsShape)));
    for (std::size_t i = 0; i < lhsShape.size() - 2; i++)
        if (lhsShape[i] != rhsShape[i])
            throw TensorInvalidShapeException(std::format("Tensors dimensions are not compatible for batched matrix "
                                                          "multiplication (use broadcasting) : {} and {}.",
                                                          Tensor<T, CppBackend>::shapeToStr(lhsShape),
                                                          Tensor<T, CppBackend>::shapeToStr(rhsShape)));

    std::size_t numel = rhsShape[0] * lhsShape[1] * rhsShape[2];
    CppBackend<T>::TStorage newStorage = CppBackend<T>::allocate(numel);

    for (std::size_t i = 0; i < numel; i++)
        for (std::size_t k = 0; k < lhsShape[lhsShape.size() - 1]; k++)
            newStorage[i] += lhsStorage[k + (i / rhsShape[2]) * lhsShape[lhsShape.size() - 1]] *
                                 rhsStorage[(i % rhsShape[2]) + k * rhsShape[rhsShape.size() - 1]];
    return newStorage;
}