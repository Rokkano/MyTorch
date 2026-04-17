#pragma once

#include "src/mt/imt.hh"
#include "src/tensor/backend/concept.hh"

#include <concepts>

template <typename T, typename B>
requires IsBackend<T, B>
class Tensor;
