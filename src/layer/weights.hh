#pragma once

#include "src/tensor/tensor_fwd.hh"

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

void initialize_weights(Tensor<int> &tensor, enum Initialization initialization = ZEROS);
void initialize_weights(Tensor<float> &tensor, enum Initialization initialization = ZEROS);

#include "weights.hxx"