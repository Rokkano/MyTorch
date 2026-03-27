#include "tensor.hh"

bool testTensorDot(std::vector<float> lhs_buffer, std::vector<float> rhs_buffer, float expected)
{
    Tensor<float> lhs = Tensor<float>(VEC{lhs_buffer.size()}, lhs_buffer);
    Tensor<float> rhs = Tensor<float>(VEC{rhs_buffer.size()}, rhs_buffer);
    Tensor<float> dot = Tensor<float>::dot(lhs, rhs);
    ASSERT(dot.shape().size() == 1, std::format("Result should have 1 dimension but got {}.",
        dot.shape().size())
    );
    ASSERT(dot.numel() == 1, std::format("Result should have 1 element but got {}.",
        dot.numel())
    );
    ASSERT(dot.item() == expected, std::format("Dot product does not match : expected {} but got {}.",
        expected,
        dot.item())
    );
    return true;
};
PARAMETRIZE(testTensorDot, std::vector<float>{1, 2, 3}, std::vector<float>{4, 5, 6}, 32.0f)
PARAMETRIZE(testTensorDot, std::vector<float>{1, 0, 0}, std::vector<float>{0, 1, 0}, 0.0f)
PARAMETRIZE(testTensorDot, std::vector<float>{1, 1, 1}, std::vector<float>{1, 1, 1}, 3.0f)
PARAMETRIZE(testTensorDot, std::vector<float>{-1, -2, -3}, std::vector<float>{1, 2, 3}, -14.0f)
PARAMETRIZE(testTensorDot, std::vector<float>{3.14f}, std::vector<float>{2.0f}, 6.28f)
PARAMETRIZE_THROW(testTensorDot, std::vector<float>{1, 2, 3}, std::vector<float>{1, 2}, 0.0f)
PARAMETRIZE_THROW(testTensorDot, std::vector<float>{1, 2}, std::vector<float>{1, 2, 3}, 0.0f)

bool testTensorDotInvalidShape(std::vector<std::size_t> lhs_shape, std::vector<std::size_t> rhs_shape)
{
    Tensor<float> lhs = Tensor<float>(lhs_shape, std::vector<float>(Tensor<float>(lhs_shape).numel(), 1.0f));
    Tensor<float> rhs = Tensor<float>(rhs_shape, std::vector<float>(Tensor<float>(rhs_shape).numel(), 1.0f));
    Tensor<float>::dot(lhs, rhs);
    return true;
};
PARAMETRIZE_THROW(testTensorDotInvalidShape, VEC{2, 3}, VEC{2, 3})
PARAMETRIZE_THROW(testTensorDotInvalidShape, VEC{2, 3}, VEC{6})
PARAMETRIZE_THROW(testTensorDotInvalidShape, VEC{6}, VEC{2, 3})

bool testTensorMM(std::vector<std::size_t> lhs_shape, std::vector<float> lhs_buffer,
                  std::vector<std::size_t> rhs_shape, std::vector<float> rhs_buffer,
                  std::vector<float> expected)
{
    Tensor<float> lhs = Tensor<float>(lhs_shape, lhs_buffer);
    Tensor<float> rhs = Tensor<float>(rhs_shape, rhs_buffer);
    Tensor<float> mm = Tensor<float>::mm(lhs, rhs);
    ASSERT(mm.shape().size() == 2, std::format("Result should have 2 dimensions but got {}.",
        mm.shape().size())
    );
    ASSERT(mm.shape()[0] == lhs_shape[0], std::format("Result rows do not match : expected {} but got {}.",
        lhs_shape[0], mm.shape()[0])
    );
    ASSERT(mm.shape()[1] == rhs_shape[1], std::format("Result cols do not match : expected {} but got {}.",
        rhs_shape[1], mm.shape()[1])
    );
    for (std::size_t i = 0; i < mm.numel(); i++)
        ASSERT(mm[i] == expected[i], std::format("Buffer value does not match at index {} : expected {} but got {}.",
            i, expected[i], mm[i])
        );
    return true;
};

// 2x2 * 2x2
PARAMETRIZE(testTensorMM,
    VEC{2, 2}, std::vector<float>{1, 2, 3, 4},
    VEC{2, 2}, std::vector<float>{5, 6, 7, 8},
    std::vector<float>{19, 22, 43, 50})

// 2x3 * 3x2
PARAMETRIZE(testTensorMM,
    VEC{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
    VEC{3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12},
    std::vector<float>{58, 64, 139, 154})

// 3x3 * 3x3 (Identity)
PARAMETRIZE(testTensorMM,
    VEC{3, 3}, std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1},
    VEC{3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9},
    std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9})

// Empty matrix
PARAMETRIZE(testTensorMM,
    VEC{2, 2}, std::vector<float>(4, 0.0f),
    VEC{2, 2}, std::vector<float>{1, 2, 3, 4},
    std::vector<float>(4, 0.0f))

// 1x3 * 3x1
PARAMETRIZE(testTensorMM,
    VEC{1, 3}, std::vector<float>{1, 2, 3},
    VEC{3, 1}, std::vector<float>{4, 5, 6},
    std::vector<float>{32})

PARAMETRIZE_THROW(testTensorMM, VEC{2, 3}, std::vector<float>(6, 1.0f), VEC{2, 3}, std::vector<float>(6, 1.0f), std::vector<float>())
PARAMETRIZE_THROW(testTensorMM, VEC{2, 3}, std::vector<float>(6, 1.0f), VEC{4, 2}, std::vector<float>(8, 1.0f), std::vector<float>())