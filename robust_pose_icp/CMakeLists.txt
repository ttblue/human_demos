cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(alignment_prerejective)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (alignment_prerejective alignment_prerejective.cpp)
target_link_libraries (alignment_prerejective ${PCL_LIBRARIES})

add_executable (discard_plane discard_plane_points.cpp)
target_link_libraries (discard_plane ${PCL_LIBRARIES})


add_executable (icp simple_icp.cpp)
target_link_libraries (icp ${PCL_LIBRARIES})
