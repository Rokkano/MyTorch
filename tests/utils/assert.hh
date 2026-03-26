#pragma once

#include <sstream>

inline auto &assertBuffer()
{
    static std::stringstream ss;
    return ss;
};

#define ASSERT_(expr)                                                                                                  \
    if (!(expr))                                                                                                       \
    {                                                                                                                  \
        assertBuffer() << "ASSERT FAILED "                                                                             \
                          "(" __FILE__ ":" TO_STRING(__LINE__) "): "                                                   \
                       << #expr;                                                                                       \
        return false;                                                                                                  \
    }

#define ASSERT_FMSG(expr, fmsg)                                                                                        \
    if (!(expr))                                                                                                       \
    {                                                                                                                  \
        assertBuffer() << "ASSERT FAILED "                                                                             \
                          "(" __FILE__ ":" TO_STRING(__LINE__) "): "                                                   \
                       << fmsg;                                                                                        \
        return false;                                                                                                  \
    }

#define ASSERT_FMSG_SMSG(expr, fmsg, smsg)                                                                             \
    if (!(expr))                                                                                                       \
    {                                                                                                                  \
        assertBuffer() << "ASSERT FAILED "                                                                             \
                          "(" __FILE__ ":" TO_STRING(__LINE__) "): "                                                   \
                       << fmsg;                                                                                        \
        return false;                                                                                                  \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        assertBuffer() << smsg;                                                                                        \
    }

#define _GET_ASSERT(_1, _2, _3, NAME, ...) NAME

#define ASSERT(...) _GET_ASSERT(__VA_ARGS__, ASSERT_FMSG_SMSG, ASSERT_FMSG, ASSERT_, )(__VA_ARGS__)
