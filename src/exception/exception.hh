#pragma once

#include <exception>
#include <string>

class Exception : public std::exception
{
protected:
    std::string msg_;

public:
    explicit Exception(const char *message)
        : msg_(message) {}

    explicit Exception(const std::string &message)
        : msg_(message) {}
    virtual ~Exception() noexcept {}

    virtual const char *what() const noexcept
    {
        return msg_.c_str();
    }
};

#define DEFINE_EXCEPTION(CLASSNAME)    \
    class CLASSNAME : public Exception \
    {                                  \
        using Exception::Exception;    \
    };

DEFINE_EXCEPTION(CastException);

#include "tensor.hh"