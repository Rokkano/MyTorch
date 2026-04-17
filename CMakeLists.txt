cmake_minimum_required(VERSION 3.15...4.0)

project(MyTorch 
    VERSION 1.0
    DESCRIPTION "A small reproduction of a functionning resnet18 from scratch in C++"
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(DCMAKE_C_COMPILER "gcc-15")
set(DCMAKE_CXX_COMPILER "g++-15")

string(ASCII 27 ESCAPE)
set(BoldWhite   "${ESCAPE}[1;37m")
set(BoldYellow  "${ESCAPE}[1;33m")
set(BoldRed     "${ESCAPE}[1;31m")
set(ColourReset "${ESCAPE}[m")

# LIBRARIES
include(FetchContent)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
if(OpenCV_FOUND)
    list(APPEND LIBRARIES ${OpenCV_LIBS})
endif()

# BUILD CONFIG
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -Wall -Werror -Wextra -fconcepts-diagnostics-depth=200 -pedantic -O3 -pg")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OPT_FLAGS} -g3 -g -rdynamic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}")

file(GLOB SRC src/*.cc src/*/*.cc src/*/*/*.cc)
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc" "${CMAKE_CURRENT_SOURCE_DIR}/src/linear.cc")

# BACKEND
if( BACKEND_EIGEN )
    message("${BoldWhite}-- Using Eigen BLAS Backend${ColorReset}")
elseif ( BACKEND_CPP )
    message("${BoldWhite}-- Using C++ Backend${ColorReset}")
else ()
    message("${BoldYellow}-- No backend specified, fallback to default C++ Backend${ColorReset}")
endif()

add_compile_definitions(BACKEND_EIGEN=$<BOOL:${BACKEND_EIGEN}>)
add_compile_definitions(BACKEND_CPP=$<BOOL:${BACKEND_CPP}>)


# TARGET : main
add_executable(main)
target_sources(main PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc" ${SRC})
target_link_libraries(main PRIVATE ${LIBRARIES} ${OpenCV_LIBS})


# TARGET : benchmark
add_executable(benchmark)
target_sources(benchmark PRIVATE "scripts/benchmark.cc" ${SRC})
target_link_libraries(benchmark PRIVATE ${LIBRARIES} ${OpenCV_LIBS})


# TARGET : test
file(GLOB SRC_TST tests/*.cc tests/*/*.cc tests/*/*/*.cc)
add_executable(test)
target_sources(test PRIVATE ${SRC} ${SRC_TST})
target_link_libraries(test PRIVATE ${LIBRARIES} ${OpenCV_LIBS})

