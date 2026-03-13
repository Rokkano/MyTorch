cmake_minimum_required(VERSION 3.15...4.0)

project(MyTorch 
    VERSION 1.0
    DESCRIPTION "A small reproduction of a functionning resnet18 from scratch in C++"
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(DCMAKE_C_COMPILER "gcc-15")
set(DCMAKE_CXX_COMPILER "g++-15")

# LIBRARIES
include(FetchContent)

# matplot++
FetchContent_Declare(matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_TAG origin/master)
FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
    FetchContent_Populate(matplotplusplus)
    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(LIBRARIES "${LIBRARIES} matplot")
string(STRIP ${LIBRARIES} LIBRARIES)

# BUILD CONFIG
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -Wall -Werror -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OPT_FLAGS} -g3 -g -rdynamic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")

include_directories("${PROJECT_SOURCE_DIR}/include")

# TARGET : main
file(GLOB SRC src/*.cc src/*/*.cc src/*/*/*.cc)
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc")

add_executable(main)
target_sources(main PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc" ${SRC})
target_link_libraries(main PRIVATE ${LIBRARIES})


# TARGET : benchmark
add_executable(benchmark)
target_sources(benchmark PRIVATE "scripts/benchmark.cc" ${SRC})
target_link_libraries(benchmark PRIVATE ${LIBRARIES})


# TARGET : test
file(GLOB SRC_TST tests/*.cc tests/*/*.cc tests/*/*/*.cc)
add_executable(test)
target_sources(test PRIVATE ${SRC} ${SRC_TST})
target_link_options(test PRIVATE "-lcriterion")
target_link_libraries(test PRIVATE ${LIBRARIES})

