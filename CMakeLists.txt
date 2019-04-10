# Library name
	set(LIB_NAME hogsvm)

# Path to libraries 
	set(INCLUDE_PATH lib)

# Load required source files in libraries
	set(LIB_SOURCES ${INCLUDE_PATH}/tinyxml2.cpp)

add_library(${LIB_NAME} STATIC hogsvm.cpp ${LIB_SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${INCLUDE_PATH})
