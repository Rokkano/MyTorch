#pragma once

#include "assert.hh"

#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

template <typename Func, typename... Args>
    requires std::same_as<std::invoke_result_t<Func, Args...>, bool>
bool run(Func func, Args &&...args)
{
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
    try
    {
        bool result = func(std::forward<Args>(args)...);
        std::cout.rdbuf(old);
        return result;
    }
    catch (const std::exception &e)
    {
        std::cout.rdbuf(old);
        assertBuffer() << "EXCEPTION CAUGHT: "
                       << e.what();
        return false;
    }
}

template <typename Func, typename... Args>
    requires std::invocable<Func, Args...>
bool run_throw(Func func, Args &&...args)
{
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
    try
    {
        func(std::forward<Args>(args)...);
        std::cout.rdbuf(old);
        return false;
    }
    catch (const std::exception &e)
    {
        std::cout.rdbuf(old);
        return true;
    }
}

template <typename Func, typename... Args>
    requires std::invocable<Func, Args...>
bool run_no_throw(Func func, Args &&...args)
{
    try
    {
        func(std::forward<Args>(args)...);
        return true;
    }
    catch (const std::exception &e)
    {
        return false;
    }
}