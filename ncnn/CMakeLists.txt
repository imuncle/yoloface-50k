cmake_minimum_required(VERSION 2.8.3)
project(yoloface)

add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)
find_package(Vulkan REQUIRED)

set(CMAKE_CXX_FLAGS "-fopenmp -lm")

include_directories(
  include/ncnn
  ${OpenCV_INCLUDE_DIRECTORIES}
)

add_executable(yoloface
  main.cpp
)

target_link_libraries(yoloface
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/libncnn.a
  m
  Vulkan::Vulkan
  ${OpenCV_LIBRARIES}
  pthread
)
