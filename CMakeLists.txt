cmake_minimum_required(VERSION 3.16)

project(math_library)

find_program(intel_compiler icpc)

if (NOT("${intel_compiler}" STREQUAL "intel_compiler-NOTFOUND"))
	message("Use Intel Compiler")
	SET(CMAKE_CXX_COMPILER "icpc")
	SET(CMAKE_CXX_FLAGS "-restrict -std=c++17 -Ofast -march=native -fopenmp ${CMAKE_CXX_FLAGS}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	message("Use GNU Compiler")
	SET(CMAKE_CXX_FLAGS "-std=c++17 -Ofast -march=native -fopenmp ${CMAKE_CXX_FLAGS}")
else()
	message(FATAL_ERROR "Only GNU Compiler and Intel compiler are supported")
endif()

file(GLOB_RECURSE LIB_DIR_LIST "src/*.cpp")
file(GLOB_RECURSE TEST_DIR_LIST "test/*.cpp")
include_directories(./include/)

add_executable(test ${TEST_DIR_LIST})
add_library(math_library STATIC ${LIB_DIR_LIST})

target_link_libraries(test math_library)

install(TARGETS math_library DESTINATION lib)