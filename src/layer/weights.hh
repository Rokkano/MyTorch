#pragma once

#include "../tensor/tensor.hh"

enum Initialization {
    ZEROS,
    ONES,
    NORMAL,
    UNIFORM,
    XAVIER,
    XAVIER_NORMALIZED,
    HE,
    he_leaky,
};

void initialize_weights(Tensor<float> &tensor, enum Initialization initialization = ZEROS);
