#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <sstream>
#include <map>

inline std::map<std::string, std::function<bool()>> REGISTRY;

#define CONCAT(a, b) a##b
#define EXPAND(a, b) CONCAT(a, b)
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define TESTFUNCNAME(func, id) EXPAND(EXPAND(func, _test), id)
#define TESTSTRUCTNAME(func, id) EXPAND(EXPAND(func, _test_struct), id)

#define ASSERT(expr) \
    if (!(expr)) return false;

#define PARAMETRIZE(func, ...) \
    PARAMETRIZE_IMPL(EXPAND(_id_, __COUNTER__), func, __VA_ARGS__)

#define PARAMETRIZE_IMPL(id, func, ...) \
bool TESTFUNCNAME(func, id)() { \
    return run(func, __VA_ARGS__); \
} \
struct TESTSTRUCTNAME(func, id) { \
    TESTSTRUCTNAME(func, id)() { \
        REGISTRY[TOSTRING(TESTFUNCNAME(func, id))] = TESTFUNCNAME(func, id); \
    } \
} EXPAND(_instance_, id);


template<typename Func, typename... Args>
requires std::same_as<std::invoke_result_t<Func, Args...>, bool>
bool run(Func func, Args&&... args)
{
    std::stringstream buffer;
    std::streambuf * old = std::cout.rdbuf(buffer.rdbuf());
    try
    {
        bool result = func(std::forward<Args>(args)...);
        std::cout.rdbuf(old);
        return result;
    }
    catch(const std::exception& e)
    {
        std::cout << buffer.str() << std::endl;
        std::cerr << e.what() << '\n';
        std::cout.rdbuf(old);
        throw;
    }
};

#define PARAMETRIZE_THROW(func, ...) \
    PARAMETRIZE_THROW_IMPL(EXPAND(_id_, __COUNTER__), func, __VA_ARGS__)

#define PARAMETRIZE_THROW_IMPL(id, func, ...) \
bool TESTFUNCNAME(func, id)() { \
    return run_throw(func, __VA_ARGS__); \
} \
struct TESTSTRUCTNAME(func, id) { \
    TESTSTRUCTNAME(func, id)() { \
        REGISTRY[TOSTRING(TESTFUNCNAME(func, id))] = TESTFUNCNAME(func, id); \
    } \
} EXPAND(_instance_, id);

template<typename Func, typename... Args>
requires std::same_as<std::invoke_result_t<Func, Args...>, bool>
bool run_throw(Func func, Args&&... args)
{
    std::stringstream buffer;
    std::streambuf * old = std::cout.rdbuf(buffer.rdbuf());
    try
    {
        func(std::forward<Args>(args)...);
        std::cout.rdbuf(old);
        return false;
    }
    catch(const std::exception& e)
    {
        std::cout.rdbuf(old);
        return true;
    }
}

#define PARAMETRIZE_NO_THROW(func, ...) \
    PARAMETRIZE_NO_THROW_IMPL(EXPAND(_id_, __COUNTER__), func, __VA_ARGS__)

#define PARAMETRIZE_NO_THROW_IMPL(id, func, ...) \
bool TESTFUNCNAME(func, id)() { \
    return run_no_throw(func, __VA_ARGS__); \
} \
struct TESTSTRUCTNAME(func, id) { \
    TESTSTRUCTNAME(func, id)() { \
        REGISTRY[TOSTRING(TESTFUNCNAME(func, id))] = TESTFUNCNAME(func, id); \
    } \
} EXPAND(_instance_, id);

template<typename Func, typename... Args>
requires std::same_as<std::invoke_result_t<Func, Args...>, bool>
bool run_no_throw(Func func, Args&&... args)
{
    try
    {
        func(std::forward<Args>(args)...);
        return true;
    }
    catch(const std::exception& e)
    {
        return false;
    }
}