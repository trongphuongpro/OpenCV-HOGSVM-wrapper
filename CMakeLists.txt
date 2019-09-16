cmake_minimum_required(VERSION 3.4)
find_package(OpenCV REQUIRED)

add_compile_options(
	-std=c++11
	-Wall
	-Werror
	-O2
)

set(LIB_NAME hogsvm)

add_library(${LIB_NAME} STATIC src/hogsvm.cpp lib/tinyxml2.cpp)
target_link_libraries(${LIB_NAME} ${OpenCV_LIBS})
target_include_directories(${LIB_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} include)