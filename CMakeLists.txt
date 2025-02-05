cmake_minimum_required (VERSION 3.14) # CMake minimum version required. It is always good to ensure!
project (PhaseFieldJr) # name of your project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/boram)

add_executable(example1 example1.cpp)
target_link_libraries(example1 PUBLIC Eigen3::Eigen)

add_executable(example2 example2.cpp)
target_link_libraries(example2 PUBLIC Eigen3::Eigen)
