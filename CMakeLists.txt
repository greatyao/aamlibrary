cmake_minimum_required(VERSION 2.6)

project (AAMLibrary)

list(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O2")

MESSAGE(STATUS "Project Directory: ${PROJECT_SOURCE_DIR}")
MESSAGE(STATUS "Project Directory: ${PROJECT_BINARY_DIR}")


find_package(OpenCV REQUIRED)

add_subdirectory(src)
add_subdirectory(exe)
