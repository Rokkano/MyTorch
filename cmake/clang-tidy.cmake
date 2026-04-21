include(cmake/color.cmake)

find_program(CLANG_TIDY "clang-tidy")

if(CLANG_TIDY)
    message(STATUS "${BoldWhite}Adding clang-tidy target to Makefile${ColorReset}")

    file(GLOB_RECURSE CLANG_TIDY_TARGETS CONFIGURE_DEPENDS 
        "${PROJECT_SOURCE_DIR}/src/*.cc" 
        "${PROJECT_SOURCE_DIR}/src/*.hh" 
        "${PROJECT_SOURCE_DIR}/src/*.hxx" 
        # "${PROJECT_SOURCE_DIR}/tests/*.cc" 
        # "${PROJECT_SOURCE_DIR}/tests/*.hh" 
        # "${PROJECT_SOURCE_DIR}/tests/*.hxx" 
    )
    add_custom_target(clang-tidy COMMAND /usr/bin/clang-tidy ${CLANG_TIDY_TARGETS} --fix -checks=-*,clang-analyzer-*,-clang-analyzer-cplusplus* -- -std=c++20 -Iinclude -I./)

else ()
    message(STATUS "${BoldYellow}clang-tidy not found${ColorReset}")
endif()