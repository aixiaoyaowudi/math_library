cmake_minimum_required(VERSION 3.1.3)

project(math_library)

find_program(intel_compiler icpc)

file(GLOB_RECURSE LIB_DIR_LIST "src/*.cpp")
file(GLOB_RECURSE TEST_DIR_LIST "test/*.cpp")
include_directories(./include/)

add_executable(func_test ${TEST_DIR_LIST})
add_library(${PROJECT_NAME} SHARED ${LIB_DIR_LIST})

if (NOT("${intel_compiler}" STREQUAL "intel_compiler-NOTFOUND"))
	message("-- Use Intel Compiler")
	SET(CMAKE_CXX_COMPILER "icpc")
	SET(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
	SET(CMAKE_CXX_FLAGS "-Ofast ${CMAKE_CXX_FLAGS}")
	SET(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
	SET(CMAKE_CXX_FLAGS "-fopenmp ${CMAKE_CXX_FLAGS}")
	SET(CMAKE_CXX_FLAGS "-flto ${CMAKE_CXX_FLAGS}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	message("-- Use GNU Compiler")
	SET(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
	SET(CMAKE_CXX_FLAGS "-Ofast ${CMAKE_CXX_FLAGS}")
	SET(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
	SET(CMAKE_CXX_FLAGS "-flto ${CMAKE_CXX_FLAGS}")
	# SET(CMAKE_CXX_FLAGS "-fsanitize=address,undefined ${CMAKE_CXX_FLAGS}")
	find_package(OpenMP REQUIRED)
	if(OpenMP_FOUND)
    	target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
	else()
    	message(FATAL_ERROR "OpenMP not found!")
	endif()
else()
	message(FATAL_ERROR "Only GNU Compiler and Intel compiler are supported")
endif()

target_link_libraries(func_test ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION /lib)
install(DIRECTORY ./include/ DESTINATION include)

if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
