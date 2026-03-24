#pragma once

#include "utils/utils.hh"

#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct Record
{
    std::function<bool()> runner;
    std::string func;
    std::string file;
    std::string line;
    std::string args;
};

inline auto &registry()
{
    static std::map<std::string, Record> _registry;
    return _registry;
};

inline auto &functionalRegistry()
{
    static std::map<std::string, Record> _functionalRegistry;
    return _functionalRegistry;
};

#define TEST_FUNC_NAME(func, id, ...) CONCAT(func, _test, id, __VA_ARGS__)
#define TEST_STRUCT_NAME(func, id, ...) CONCAT(func, _test_struct, id, __VA_ARGS__)
#define TEST_STRUCT_INSTANCE_NAME(func, id, ...) CONCAT(func, _test_struct_instance, id, __VA_ARGS__)

#define PARAMETRIZE(func, ...) PARAMETRIZE_IMPL(CONCAT(_id_, __COUNTER__), func, run, registry(), __VA_ARGS__)

#define PARAMETRIZE_FUNCTIONAL(func, ...)                                                                              \
    PARAMETRIZE_IMPL(CONCAT(_id_, __COUNTER__), func, run, functionalRegistry(), __VA_ARGS__)

#define PARAMETRIZE_THROW(func, ...)                                                                                   \
    PARAMETRIZE_IMPL(CONCAT(_id_, __COUNTER__), func, run_throw, registry(), __VA_ARGS__)

#define PARAMETRIZE_NO_THROW(func, ...)                                                                                \
    PARAMETRIZE_IMPL(CONCAT(_id_, __COUNTER__), func, run_no_throw, registry(), __VA_ARGS__)

#define PARAMETRIZE_IMPL(id, func, runner_, registry_, ...)                                                            \
    bool TEST_FUNC_NAME(func, id)() { return runner_(func __VA_OPT__(, ) __VA_ARGS__); }                               \
    struct TEST_STRUCT_NAME(func, id)                                                                                  \
    {                                                                                                                  \
        TEST_STRUCT_NAME(func, id)()                                                                                   \
        {                                                                                                              \
            registry_[TO_STRING(TEST_FUNC_NAME(func, id))] = {TEST_FUNC_NAME(func, id), TO_STRING(func),               \
                                                              TO_STRING(__FILE__), TO_STRING(__LINE__),                \
                                                              std::string(#__VA_ARGS__)};                              \
        }                                                                                                              \
    } TEST_STRUCT_INSTANCE_NAME(func, id);
