cmake_minimum_required(VERSION 3.0.2)
project(NanoDet_demo)

add_compile_options(-std=c++11)
find_package(OpenCV REQUIRED)

set(CMAKE_BUILD_TYPE DEBUG)


INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_SOURCE_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# AUX_SOURCE_DIRECTORY(src DIR_SRCS)
ADD_EXECUTABLE(NanoDet_demo src/main.cpp src/NanoDet.cpp src/tengine_operations.c src/timer.cc)

target_link_libraries(NanoDet_demo tengine-lite ${OpenCV_LIBS})
