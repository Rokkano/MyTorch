include(cmake/color.cmake)

find_program(CLANG_FORMAT "clang-format")

if(CLANG_FORMAT)
    message(STATUS "${BoldWhite}Adding clang-format target to Makefile${ColorReset}")
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES *.cpp *.cxx *.cc *.hh *.hxx)
    add_custom_target(clang-format COMMAND /usr/bin/clang-format -i -style=file ${ALL_CXX_SOURCE_FILES})
else ()
    message(STATUS "${BoldYellow}clang-format not found${ColorReset}")
endif()