# Library name
set(LIB_NAME hogsvm)

add_library(${LIB_NAME} STATIC src/hogsvm.cpp lib/tinyxml2.cpp)

target_include_directories(${LIB_NAME} PUBLIC include)
