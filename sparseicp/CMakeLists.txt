cmake_minimum_required(VERSION 2.4.6)
project(sparseicp)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(Eigen3 REQUIRED)
add_definitions("-DEIGEN_DEFAULT_TO_ROW_MAJOR")
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (try try.cpp)
target_link_libraries (try ${PCL_LIBRARIES})
