cmake_minimum_required(VERSION 3.16)
project(ukf_tracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(ukf
    src/ukf.cpp
    src/pose_utils.cpp
)

target_include_directories(ukf PUBLIC include)
target_link_libraries(ukf PUBLIC Eigen3::Eigen)

add_executable(ukf_demo tests/demo.cpp)
target_link_libraries(ukf_demo PRIVATE ukf)

add_executable(ukf_smoke tests/smoke.cpp)
target_link_libraries(ukf_smoke PRIVATE ukf)
