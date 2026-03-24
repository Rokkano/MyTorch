#pragma once

#include "assert.hh"
#include "macro.hh"

template <typename Func, typename... Args>
    requires std::same_as<std::invoke_result_t<Func, Args...>, bool>
bool run(Func func, Args &&...args);

template <typename Func, typename... Args>
    requires std::invocable<Func, Args...>
bool run_throw(Func func, Args &&...args);

template <typename Func, typename... Args>
    requires std::invocable<Func, Args...>
bool run_no_throw(Func func, Args &&...args);

#include "runner.hxx"