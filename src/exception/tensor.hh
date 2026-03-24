#pragma once

#include "exception.hh"

#include <exception>
#include <string>

DEFINE_EXCEPTION(TensorOutOfBoundException);
DEFINE_EXCEPTION(TensorInvalidShapeException);
DEFINE_EXCEPTION(TensorInvalidTypeException);
DEFINE_EXCEPTION(TensorBroadcastException);
DEFINE_EXCEPTION(TensorSqueezeException);
DEFINE_EXCEPTION(TensorTransposeException);
