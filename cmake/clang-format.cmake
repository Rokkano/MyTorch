include(cmake/color.cmake)

find_program(CLANG_FORMAT "clang-format")

if(CLANG_FORMAT)
    message(STATUS "${BoldWhite}Adding clang-format target to Makefile${ColorReset}")

    file(GLOB_RECURSE CLANG_FORMAT_TARGETS CONFIGURE_DEPENDS 
        "${PROJECT_SOURCE_DIR}/src/*.cc" 
        "${PROJECT_SOURCE_DIR}/src/*.hh" 
        "${PROJECT_SOURCE_DIR}/src/*.hxx" 
        "${PROJECT_SOURCE_DIR}/tests/*.cc" 
        "${PROJECT_SOURCE_DIR}/tests/*.hh" 
        "${PROJECT_SOURCE_DIR}/tests/*.hxx" 
    )
    add_custom_target(clang-format COMMAND /usr/bin/clang-format -i -style=file ${CLANG_FORMAT_TARGETS})

else ()
    message(STATUS "${BoldYellow}clang-format not found${ColorReset}")
endif()