#include "tensor.hh"

bool testTensorCreation()
{
    Tensor<float> tensor = Tensor<float>();
    ASSERT(tensor.shape()[0] == 0)
    return true;
}
PARAMETRIZE(testTensorCreation)

bool testTensorCreationWithShape(std::vector<std::size_t> shape)
{
    Tensor<float> tensor = Tensor<float>(shape);
    for (std::size_t i = 0; i < shape.size(); i++)
        ASSERT(tensor.shape()[i] == shape[i], std::format("Shape does not match : {} and {}.", 
            Tensor<float>::tensorShapeToStr(tensor.shape()), 
            Tensor<float>::tensorShapeToStr(shape))
        );
    return true;
};
PARAMETRIZE(testTensorCreationWithShape, VEC{128})
PARAMETRIZE(testTensorCreationWithShape, VEC{2, 40})
PARAMETRIZE(testTensorCreationWithShape, VEC{1, 1, 1, 1})
PARAMETRIZE(testTensorCreationWithShape, VEC{1, 2, 20, 3})


bool testTensorCreationWithShapeAndBuffer(std::vector<std::size_t> shape, std::vector<float> buffer)
{
    Tensor<float> tensor = Tensor<float>(shape, buffer);
    for (std::size_t i = 0; i < shape.size(); i++)
        ASSERT(tensor.shape()[i] == shape[i], std::format("Shape does not match : expected {} but got {}.", 
            Tensor<float>::tensorShapeToStr(shape), 
            Tensor<float>::tensorShapeToStr(tensor.shape()))
        );
    return true;
};
PARAMETRIZE(testTensorCreationWithShapeAndBuffer, VEC{128}, std::vector<float>(128))
PARAMETRIZE(testTensorCreationWithShapeAndBuffer, VEC{2, 40}, std::vector<float>(80))
PARAMETRIZE(testTensorCreationWithShapeAndBuffer, VEC{1, 1, 1, 1}, std::vector<float>(1))
PARAMETRIZE(testTensorCreationWithShapeAndBuffer, VEC{1, 2, 20, 3}, std::vector<float>(2 * 20 * 3))
PARAMETRIZE_THROW(testTensorCreationWithShapeAndBuffer, VEC{128}, std::vector<float>(0))
PARAMETRIZE_THROW(testTensorCreationWithShapeAndBuffer, VEC{128}, std::vector<float>(127))
PARAMETRIZE_THROW(testTensorCreationWithShapeAndBuffer, VEC{1, 2, 20, 3}, std::vector<float>(2 * 20 * 4))


bool testCoordToAbs(std::vector<std::size_t> shape, std::vector<std::size_t> coordinates, std::size_t expected)
{
    Tensor<float> tensor = Tensor<float>(shape);
    std::size_t abs = tensor.coordToAbs(coordinates);
    ASSERT(abs == expected, std::format("Coords does not match : expected {} but got {}.", expected, abs));
    return true;
};
PARAMETRIZE_NO_THROW(testCoordToAbs, VEC{128}, VEC{0}, 0)
PARAMETRIZE_NO_THROW(testCoordToAbs, VEC{128}, VEC{127}, 127)
PARAMETRIZE_THROW(testCoordToAbs, VEC{128}, VEC{128}, 0)
PARAMETRIZE_THROW(testCoordToAbs, VEC{128}, VEC{127, 1}, 0)
PARAMETRIZE_NO_THROW(testCoordToAbs, VEC{2, 5}, VEC{0, 0}, 0)
PARAMETRIZE_NO_THROW(testCoordToAbs, VEC{2, 5}, VEC{1, 4}, 9)
PARAMETRIZE_THROW(testCoordToAbs, VEC{2, 5}, VEC{127}, 0)
PARAMETRIZE_THROW(testCoordToAbs, VEC{2, 5}, VEC{2, 5}, 0)
PARAMETRIZE(testCoordToAbs, VEC{128}, VEC{0}, 0)
PARAMETRIZE(testCoordToAbs, VEC{128}, VEC{127}, 127)
PARAMETRIZE(testCoordToAbs, VEC{128}, VEC{64}, 64)
PARAMETRIZE(testCoordToAbs, VEC{2, 5}, VEC{0, 0}, 0)
PARAMETRIZE(testCoordToAbs, VEC{2, 5}, VEC{0, 3}, 3)
PARAMETRIZE(testCoordToAbs, VEC{3, 256, 256}, VEC{0, 0, 0}, 0)
PARAMETRIZE(testCoordToAbs, VEC{3, 256, 256}, VEC{0, 255, 0}, 255 * 256)
PARAMETRIZE(testCoordToAbs, VEC{3, 256, 256}, VEC{0, 54, 37}, 54 * 256 + 37)
PARAMETRIZE(testCoordToAbs, VEC{3, 256, 256}, VEC{1, 129, 49}, 1 * 256 * 256 + 129 * 256 + 49)
PARAMETRIZE(testCoordToAbs, VEC{3, 256, 256}, VEC{2, 0, 37}, 2 * 256 * 256 + 37)


bool testAbsToCoord(std::vector<std::size_t> shape, std::size_t absolute, std::vector<std::size_t> expected)
{
    Tensor<float> tensor = Tensor<float>(shape);
    std::vector<std::size_t> coord = tensor.absToCoord(absolute);

    for (std::size_t i = 0; i < coord.size(); i++)
        ASSERT(coord[i] == expected[i], std::format("Coords does not match : expected {} but got {}.", 
            Tensor<float>::tensorShapeToStr(expected), 
            Tensor<float>::tensorShapeToStr(coord))
        );
    return true;
};
PARAMETRIZE_NO_THROW(testAbsToCoord, VEC{128}, 0, VEC{0})
PARAMETRIZE_NO_THROW(testAbsToCoord, VEC{128}, 127, VEC{127})
PARAMETRIZE_THROW(testAbsToCoord, VEC{128}, 128, VEC{0})
PARAMETRIZE_NO_THROW(testAbsToCoord, VEC{2, 5}, 0, VEC{0, 0})
PARAMETRIZE_NO_THROW(testAbsToCoord, VEC{2, 5}, 9, VEC{1, 4})
PARAMETRIZE_THROW(testAbsToCoord, VEC{2, 5}, 11, VEC{2, 5})
PARAMETRIZE(testAbsToCoord, VEC{128}, 0, VEC{0})
PARAMETRIZE(testAbsToCoord, VEC{128}, 127, VEC{127})
PARAMETRIZE(testAbsToCoord, VEC{128}, 64, VEC{64})
PARAMETRIZE(testAbsToCoord, VEC{2, 5}, 0, VEC{0, 0})
PARAMETRIZE(testAbsToCoord, VEC{2, 5}, 3, VEC{0, 3})
PARAMETRIZE(testAbsToCoord, VEC{3, 256, 256}, 0, VEC{0, 0, 0})
PARAMETRIZE(testAbsToCoord, VEC{3, 256, 256}, 255 * 256, VEC{0, 255, 0})
PARAMETRIZE(testAbsToCoord, VEC{3, 256, 256}, 54 * 256 + 37, VEC{0, 54, 37})
PARAMETRIZE(testAbsToCoord, VEC{3, 256, 256}, 1 * 256 * 256 + 129 * 256 + 49, VEC{1, 129, 49})
PARAMETRIZE(testAbsToCoord, VEC{3, 256, 256}, 2 * 256 * 256 + 37, VEC{2, 0, 37})

bool testTensorFill(std::vector<std::size_t> shape, float value, float expected)
{
    Tensor<float> tensor = Tensor<float>(shape);
    tensor.fill(value);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        ASSERT(tensor[i] == value, std::format("Fill value does not match : expected {} but got {}.",
            expected,
            tensor[i])
        );
    return true;
};
PARAMETRIZE(testTensorFill, VEC{128}, 0.0f, 0.0f)
PARAMETRIZE(testTensorFill, VEC{2, 40}, 1.0f, 1.0f)
PARAMETRIZE(testTensorFill, VEC{1, 1, 1, 1}, -1.0f, -1.0f)
PARAMETRIZE(testTensorFill, VEC{1, 2, 20, 3}, 3.14f, 3.14f)

bool testTensorItem(std::vector<std::size_t> shape, float value, float expected)
{
    Tensor<float> tensor = Tensor<float>(shape);
    tensor.fill(value);
    ASSERT(tensor.item() == value, std::format("Item value does not match : expected {} but got {}.",
        expected,
        tensor.item())
    );
    return true;
};
PARAMETRIZE(testTensorItem, VEC{1}, 0.0f, 0.0f)
PARAMETRIZE(testTensorItem, VEC{1}, 1.0f, 1.0f)
PARAMETRIZE(testTensorItem, VEC{1}, -1.0f, -1.0f)
PARAMETRIZE(testTensorItem, VEC{1}, 3.14f, 3.14f)
PARAMETRIZE(testTensorItem, VEC{1, 1, 1, 1}, 42.0f, 42.0f)
PARAMETRIZE_THROW(testTensorItem, VEC{2}, 0.0f, 0.0f)
PARAMETRIZE_THROW(testTensorItem, VEC{1, 2}, 0.0f, 0.0f)
PARAMETRIZE_THROW(testTensorItem, VEC{128}, 0.0f, 0.0f)

bool testTensorNumel(std::vector<std::size_t> shape, std::size_t expected)
{
    Tensor<float> tensor = Tensor<float>(shape);
    ASSERT(tensor.numel() == expected, std::format("Numel does not match : expected {} but got {}.",
        expected,
        tensor.numel())
    );
    return true;
};
PARAMETRIZE(testTensorNumel, VEC{128}, 128)
PARAMETRIZE(testTensorNumel, VEC{2, 40}, 80)
PARAMETRIZE(testTensorNumel, VEC{1, 1, 1, 1}, 1)
PARAMETRIZE(testTensorNumel, VEC{1, 2, 20, 3}, 120)
PARAMETRIZE(testTensorNumel, VEC{}, 0)

bool testTensorFlatten(std::vector<std::size_t> shape)
{
    Tensor<float> tensor = Tensor<float>::from_function(lin, shape);
    Tensor<float> flattened = tensor.flatten();

    ASSERT(flattened.shape().size() == 1, 
        std::format("Flattened tensor should have 1 dimension but got {}.", flattened.shape().size())
    );
    ASSERT(flattened.numel() == tensor.numel(), 
        std::format("Numel does not match : expected {} but got {}.", tensor.numel(), flattened.numel())
    );
    for (std::size_t i = 0; i < flattened.numel(); i++)
        ASSERT(flattened[i] == lin(i),
            std::format("Buffer value does not match at index {} : expected {} but got {}.", i, lin(i), flattened[i])
        );
    return true;
};
PARAMETRIZE(testTensorFlatten, VEC{128})
PARAMETRIZE(testTensorFlatten, VEC{2, 40})
PARAMETRIZE(testTensorFlatten, VEC{1, 1, 1, 1})
PARAMETRIZE(testTensorFlatten, VEC{1, 2, 20, 3})

bool testTensorUnsqueeze(std::vector<std::size_t> shape, std::size_t dim, std::vector<std::size_t> expected)
{
    Tensor<float> tensor = Tensor<float>::from_function(lin, shape);
    Tensor<float> unsqueezed = tensor.unsqueeze(dim);

    ASSERT(_vectorEQ<std::size_t>(expected, unsqueezed.shape()), std::format("Shape does not match : expected {} but got {}.",
        Tensor<float>::tensorShapeToStr(expected),
        Tensor<float>::tensorShapeToStr(unsqueezed.shape()))
    );

    ASSERT(unsqueezed.numel() == tensor.numel(), std::format("Numel does not match : expected {} but got {}.",
        tensor.numel(),
        unsqueezed.numel())
    );
    for (std::size_t i = 0; i < unsqueezed.numel(); i++)
        ASSERT(unsqueezed[i] == lin(i), std::format("Buffer value does not match at index {} : expected {} but got {}.",
            i, lin(i), unsqueezed[i])
        );
    return true;
};
PARAMETRIZE(testTensorUnsqueeze, VEC{128}, 0, VEC{1, 128})
PARAMETRIZE(testTensorUnsqueeze, VEC{128}, 1, VEC{128, 1})
PARAMETRIZE(testTensorUnsqueeze, VEC{2, 40}, 0, VEC{1, 2, 40})
PARAMETRIZE(testTensorUnsqueeze, VEC{2, 40}, 1, VEC{2, 1, 40})
PARAMETRIZE(testTensorUnsqueeze, VEC{2, 40}, 2, VEC{2, 40, 1})
PARAMETRIZE(testTensorUnsqueeze, VEC{1, 2, 20, 3}, 2, VEC{1, 2, 1, 20, 3})

bool testTensorSqueeze(std::vector<std::size_t> shape, std::size_t dim, std::vector<std::size_t> expected)
{
    Tensor<float> tensor = Tensor<float>::from_function(lin, shape);
    Tensor<float> squeezed = tensor.squeeze(dim);
    
    ASSERT(_vectorEQ<std::size_t>(expected, squeezed.shape()), std::format("Shape does not match : expected {} but got {}.",
        Tensor<float>::tensorShapeToStr(expected),
        Tensor<float>::tensorShapeToStr(squeezed.shape()))
    );

    ASSERT(squeezed.numel() == tensor.numel(), std::format("Numel does not match : expected {} but got {}.",
        tensor.numel(),
        squeezed.numel())
    );
    for (std::size_t i = 0; i < squeezed.numel(); i++)
        ASSERT(squeezed[i] == lin(i), std::format("Buffer value does not match at index {} : expected {} but got {}.",
            i, lin(i), squeezed[i])
        );
    return true;
};
PARAMETRIZE(testTensorSqueeze, VEC{1, 128}, 0, VEC{128})
PARAMETRIZE(testTensorSqueeze, VEC{128, 1}, 1, VEC{128})
PARAMETRIZE(testTensorSqueeze, VEC{1, 2, 40}, 0, VEC{2, 40})
PARAMETRIZE(testTensorSqueeze, VEC{2, 1, 40}, 1, VEC{2, 40})
PARAMETRIZE(testTensorSqueeze, VEC{1, 2, 1, 20, 3}, 2, VEC{1, 2, 20, 3})
PARAMETRIZE_THROW(testTensorSqueeze, VEC{2, 40}, 0, VEC{})
PARAMETRIZE_THROW(testTensorSqueeze, VEC{1, 2, 20, 3}, 1, VEC{})


bool testTensorTranspose(std::vector<std::size_t> shape, std::size_t dim0, std::size_t dim1, std::vector<std::size_t> expected)
{
    Tensor<float> tensor = Tensor<float>::from_function(lin, shape);
    Tensor<float> transposed = tensor.transpose(dim0, dim1);
    
    ASSERT(_vectorEQ<std::size_t>(expected, transposed.shape()), std::format("Shape does not match : expected {} but got {}.",
        Tensor<float>::tensorShapeToStr(expected),
        Tensor<float>::tensorShapeToStr(transposed.shape()))
    );

    ASSERT(transposed.numel() == tensor.numel(), std::format("Numel does not match : expected {} but got {}.",
        tensor.numel(),
        transposed.numel())
    );
    for (std::size_t i = 0; i < tensor.numel(); i++)
    {
        std::vector<std::size_t> coord = tensor.absToCoord(i);
        std::swap(coord[dim0], coord[dim1]);
        ASSERT(transposed[transposed.coordToAbs(coord)] == lin(i), std::format("Buffer value does not match at index {} : expected {} but got {}.",
            i, lin(i), transposed[transposed.coordToAbs(coord)])
        );
    }
    return true;
};
PARAMETRIZE(testTensorTranspose, VEC{2, 3}, 0, 1, VEC{3, 2})
PARAMETRIZE(testTensorTranspose, VEC{4, 5}, 0, 1, VEC{5, 4})
PARAMETRIZE(testTensorTranspose, VEC{2, 3, 4}, 0, 1, VEC{3, 2, 4})
PARAMETRIZE(testTensorTranspose, VEC{2, 3, 4}, 0, 2, VEC{4, 3, 2})
PARAMETRIZE(testTensorTranspose, VEC{2, 3, 4}, 1, 2, VEC{2, 4, 3})
PARAMETRIZE(testTensorTranspose, VEC{1, 2, 3, 4}, 1, 3, VEC{1, 4, 3, 2})
PARAMETRIZE_THROW(testTensorTranspose, VEC{128}, 0, 0, VEC{})
PARAMETRIZE_THROW(testTensorTranspose, VEC{}, 0, 0, VEC{})