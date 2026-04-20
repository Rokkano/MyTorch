cmake_minimum_required(VERSION 3.15...4.0)

project(MyTorch 
    VERSION 1.0
    DESCRIPTION "A small reproduction of a functionning resnet18 from scratch in C++"
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LIBRARIES
include(FetchContent)

include(cmake/color.cmake)
include(cmake/opencv.cmake)
include(cmake/clang-format.cmake)
include(cmake/clang-tidy.cmake)

# BUILD CONFIG
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Debug")
    set(CMAKE_BUILD_TYPE "Debug")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OPT_FLAGS} -g3 -g -rdynamic -fsanitize=address -ldl -Wall -Werror -Wextra -fconcepts-diagnostics-depth=200 -pedantic -O3 -pg")

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}")

file(GLOB SRC src/*.cc src/*/*.cc src/*/*/*.cc)
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc" "${CMAKE_CURRENT_SOURCE_DIR}/src/linear.cc")

# TARGET : main
add_executable(main-cpp)
target_sources(main-cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc" ${SRC})
target_link_libraries(main-cpp PRIVATE ${LIBRARIES} ${OpenCV_LIBS})
target_compile_definitions(main-cpp PRIVATE BACKEND_CPP)

add_executable(main-eigen)
target_sources(main-eigen PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc" ${SRC})
target_link_libraries(main-eigen PRIVATE ${LIBRARIES} ${OpenCV_LIBS})
target_compile_definitions(main-eigen PRIVATE BACKEND_EIGEN)

# TARGET : benchmark
add_executable(benchmark)
target_sources(benchmark PRIVATE "scripts/benchmark.cc" ${SRC})
target_link_libraries(benchmark PRIVATE ${LIBRARIES} ${OpenCV_LIBS})

# TARGET : test
file(GLOB SRC_TST tests/*.cc tests/*/*.cc tests/*/*/*.cc)
add_executable(test)
target_sources(test PRIVATE ${SRC} ${SRC_TST})
target_link_libraries(test PRIVATE ${LIBRARIES} ${OpenCV_LIBS})

# TARGET : clean-all
add_custom_target(clean-all COMMAND rm -rf CMakeFiles/ CMakeCache.txt cmake_install.cmake Makefile build main test gmon.out main-cpp main-eigen)