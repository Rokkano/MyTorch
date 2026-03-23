#pragma once

#define _CONCAT2(a, b)                                                          a##b
#define _CONCAT3(a, b, c)                                                       a##b##c
#define _CONCAT4(a, b, c, d)                                                    a##b##c##d
#define _CONCAT5(a, b, c, d, e)                                                 a##b##c##d##e
#define _CONCAT6(a, b, c, d, e, f)                                              a##b##c##d##e##f
#define _CONCAT7(a, b, c, d, e, f, g)                                           a##b##c##d##e##f##g
#define _CONCAT8(a, b, c, d, e, f, g, h)                                        a##b##c##d##e##f##g##h
#define _CONCAT9(a, b, c, d, e, f, g, h, i)                                     a##b##c##d##e##f##g##h##i
#define _CONCAT10(a, b, c, d, e, f, g, h, i, j)                                 a##b##c##d##e##f##g##h##i##j
#define _CONCAT11(a, b, c, d, e, f, g, h, i, j, k)                              a##b##c##d##e##f##g##h##i##j##k
#define _CONCAT12(a, b, c, d, e, f, g, h, i, j, k, l)                           a##b##c##d##e##f##g##h##i##j##k##l
#define _CONCAT13(a, b, c, d, e, f, g, h, i, j, k, l, m)                        a##b##c##d##e##f##g##h##i##j##k##l##m
#define _CONCAT14(a, b, c, d, e, f, g, h, i, j, k, l, m, n)                     a##b##c##d##e##f##g##h##i##j##k##l##m##n
#define _CONCAT15(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o)                  a##b##c##d##e##f##g##h##i##j##k##l##m##n##o
#define _CONCAT16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)               a##b##c##d##e##f##g##h##i##j##k##l##m##n##o##p
#define _CONCAT17(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q)            a##b##c##d##e##f##g##h##i##j##k##l##m##n##o##p##q
#define _CONCAT18(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r)         a##b##c##d##e##f##g##h##i##j##k##l##m##n##o##p##q##r
#define _CONCAT19(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s)      a##b##c##d##e##f##g##h##i##j##k##l##m##n##o##p##q##r##s
#define _CONCAT20(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t)   a##b##c##d##e##f##g##h##i##j##k##l##m##n##o##p##q##r##s##t

#define _GET_CONCAT(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, NAME, ...) NAME

#define CONCAT(...) _GET_CONCAT(__VA_ARGS__, \
    _CONCAT20,                               \
    _CONCAT19,                               \
    _CONCAT18,                               \
    _CONCAT17,                               \
    _CONCAT16,                               \
    _CONCAT15,                               \
    _CONCAT14,                               \
    _CONCAT13,                               \
    _CONCAT12,                               \
    _CONCAT11,                               \
    _CONCAT10,                               \
    _CONCAT9,                               \
    _CONCAT8,                               \
    _CONCAT7,                               \
    _CONCAT6,                               \
    _CONCAT5,                               \
    _CONCAT4,                               \
    _CONCAT3,                               \
    _CONCAT2                                \
)(__VA_ARGS__)

#define _TO_STRING(x) #x
#define TO_STRING(x) _TO_STRING(x)