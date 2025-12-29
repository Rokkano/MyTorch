#pragma once

#include <criterion/criterion.h>
#include <functional>
#include <optional>

template <typename F, typename... A>
void assert_no_throw(F &&f, A &&...args)
{
    try
    {
        std::invoke(std::forward<F>(f), std::forward<A>(args)...);
    }
    catch (const std::exception &)
    {
        do
        {
            bool cr_passed__ = !!(0);
            char *cr_msg__ = __null;
            int cr_shifted__ = 0;
            if (!cr_passed__ || criterion_options.full_stats)
            {
                do
                {
                    char *cr_def_msg__ = cr_translate_assert_msg(CRITERION_ASSERT_MSG_ANY_THROW, "");
                    char *cr_fmt_msg__ = __null;
                    cr_asprintf(&cr_fmt_msg__, "x");
                    if (cr_fmt_msg__ && cr_fmt_msg__[1])
                    {
                        cr_msg__ = cr_fmt_msg__ + 1;
                        cr_shifted__ = 1;
                        cr_asprintf_free(cr_def_msg__);
                    }
                    else
                    {
                        cr_msg__ = cr_def_msg__;
                        cr_asprintf_free(cr_fmt_msg__);
                    }
                } while (0);
                struct criterion_assert_stats cr_stat__;
                cr_stat__.passed = cr_passed__;
                cr_stat__.file = __FILE__;
                cr_stat__.line = __LINE__;
                cr_stat__.message = cr_msg__;
                criterion_send_assert(&cr_stat__);
                cr_asprintf_free(cr_msg__ - cr_shifted__);
            }
            if (!cr_passed__)
                criterion_abort_test();
            else
                cri_asserts_passed_incr();
        } while (0);
    }
}

template <typename F, typename... A>
void assert_throw(F &&f, A &&...args)
{
    try
    {
        std::invoke(std::forward<F>(f), std::forward<A>(args)...);
        do
        {
            bool cr_passed__ = !!(0);
            char *cr_msg__ = __null;
            int cr_shifted__ = 0;
            if (!cr_passed__ || criterion_options.full_stats)
            {
                do
                {
                    char *cr_def_msg__ = cr_translate_assert_msg(CRITERION_ASSERT_MSG_ANY_THROW, ""
                                                                                                 "");
                    char *cr_fmt_msg__ = __null;
                    cr_asprintf(&cr_fmt_msg__, "x");
                    if (cr_fmt_msg__ && cr_fmt_msg__[1])
                    {
                        cr_msg__ = cr_fmt_msg__ + 1;
                        cr_shifted__ = 1;
                        cr_asprintf_free(cr_def_msg__);
                    }
                    else
                    {
                        cr_msg__ = cr_def_msg__;
                        cr_asprintf_free(cr_fmt_msg__);
                    }
                } while (0);
                struct criterion_assert_stats cr_stat__;
                cr_stat__.passed = cr_passed__;
                cr_stat__.file = __FILE__;
                cr_stat__.line = __LINE__;
                cr_stat__.message = cr_msg__;
                criterion_send_assert(&cr_stat__);
                cr_asprintf_free(cr_msg__ - cr_shifted__);
            }
            if (!cr_passed__)
                criterion_abort_test();
            else
                cri_asserts_passed_incr();
        } while (0);
    }
    catch (...)
    {
    }
}

void test_throw(std::function<void()> func, bool throw_expected)
{
    if (throw_expected)
        cr_assert_any_throw(func());
    if (!throw_expected)
        cr_assert_no_throw(func());
}