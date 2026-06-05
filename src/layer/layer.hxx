#pragma once

#include "layer.hh"

template <typename T, template <typename> typename B>
void Layer<T, B>::training(bool flag)
{
    this->training_ = flag;
}

template <typename T, template <typename> typename B>
bool Layer<T, B>::training() const
{
    return this->training_;
}