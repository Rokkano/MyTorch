#pragma once

#include "src/tensor/tensor_fwd.hh"

#include <format>
#include <string_view>

enum Initialization
{
    ZEROS,
    ONES,
    NORMAL,
    UNIFORM,
    LECUN,
    XAVIER,
    HE,
};

template <>
struct std::formatter<Initialization> : std::formatter<std::string_view>
{
    auto format(Initialization c, std::format_context &ctx) const
    {
        std::string_view name;
        switch (c)
        {
        case Initialization::ZEROS:
            name = "Zeros";
            break;
        case Initialization::ONES:
            name = "Ones";
            break;
        case Initialization::NORMAL:
            name = "Normal";
            break;
        case Initialization::UNIFORM:
            name = "Uniform";
            break;
        case Initialization::LECUN:
            name = "LeCun";
            break;
        case Initialization::XAVIER:
            name = "Xavier/Glorot";
            break;
        case Initialization::HE:
            name = "HE";
            break;
        default:
            name = "Unknown";
        }
        return std::formatter<std::string_view>::format(name, ctx);
    }
};

template <template <typename> typename B>
void initialize_weights(Tensor<int, B> &tensor, enum Initialization initialization = ZEROS);

template <template <typename> typename B>
void initialize_weights(Tensor<float, B> &tensor, enum Initialization initialization = ZEROS);

#include "weights.hxx"