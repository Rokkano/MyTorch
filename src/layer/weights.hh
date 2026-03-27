#pragma once

#include "../tensor/tensor.hh"

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

void initialize_weights(Tensor<float> &tensor, enum Initialization initialization = ZEROS);
