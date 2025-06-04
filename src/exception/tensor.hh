#pragma once

#include <exception>
#include <string>

#include "exception.hh"

DEFINE_EXCEPTION(TensorOutOfBoundException);
DEFINE_EXCEPTION(TensorInvalidShapeException);
DEFINE_EXCEPTION(TensorBroadcastException);
DEFINE_EXCEPTION(TensorSqueezeException);
DEFINE_EXCEPTION(TensorTransposeException);
