#pragma once

#include "layer.hh"

template <typename T, typename B>
void Layer<T, B>::training(bool flag)
{
    this->training_ = flag;
}