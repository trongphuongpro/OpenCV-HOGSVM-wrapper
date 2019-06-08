
# Library name
set(LIB_NAME hogsvm)

# Path to libraries 
set(INCLUDE_DIRS lib)

# Load required source files in libraries
set(INCLUDE_SOURCES ${INCLUDE_DIRS}/tinyxml2.cpp)

add_library(${LIB_NAME} STATIC hogsvm.cpp ${INCLUDE_SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${INCLUDE_DIRS})
