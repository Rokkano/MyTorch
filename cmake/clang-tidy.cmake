include(cmake/color.cmake)

find_program(CLANG_TIDY "clang-tidy")

if(CLANG_TIDY)
    message(STATUS "${BoldWhite}Adding clang-tidy target to Makefile${ColorReset}")
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES *.cpp *.cxx *.cc *.hh *.hxx)
    add_custom_target(clang-tidy COMMAND /usr/bin/clang-tidy ${ALL_CXX_SOURCE_FILES} -config='' -- -std=c++20 ${INCLUDE_DIRECTORIES})
else ()
    message(STATUS "${BoldYellow}clang-tidy not found${ColorReset}")
endif()