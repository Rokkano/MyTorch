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
    XAVIER_NORMALIZED,
    HE,
};

void initialize_weights(Tensor<float> &tensor, enum Initialization initialization = ZEROS);
