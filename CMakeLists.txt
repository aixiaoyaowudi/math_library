cmake_minimum_required(VERSION 3.14.7)

project(xmal)
set(PROJECT_PYTHON_NAME PyXmal)
set(TEST_NAME func_test)

find_program(intel_compiler icpc)
find_package(SWIG REQUIRED COMPONENTS python)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)

include(UseSWIG)

file(GLOB_RECURSE LIB_DIR_LIST "${PROJECT_SOURCE_DIR}/src/*.cpp")
list(FILTER LIB_DIR_LIST EXCLUDE REGEX ".*\.origin\.cpp")
file(GLOB_RECURSE TEST_DIR_LIST "${PROJECT_SOURCE_DIR}/test/*.cpp")
list(FILTER TEST_DIR_LIST EXCLUDE REGEX ".*\.origin\.cpp")
file(GLOB_RECURSE PYTHON_DIR_LIST "${PROJECT_SOURCE_DIR}/python/src/*.cpp")
list(FILTER PYTHON_DIR_LIST EXCLUDE REGEX ".*\.origin\.cpp")

set(CMAKE_CXX_STANDARD 17)

add_executable(${TEST_NAME} ${TEST_DIR_LIST})
add_library(${PROJECT_NAME} SHARED ${LIB_DIR_LIST})

set_property(SOURCE ${PROJECT_SOURCE_DIR}/python/include/math_library_interface.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE ${PROJECT_SOURCE_DIR}/python/include/math_library_interface.i PROPERTY SWIG_MODULE_NAME ${PROJECT_PYTHON_NAME})

swig_add_library(${PROJECT_PYTHON_NAME}
	TYPE MODULE
	LANGUAGE python
	OUTPUT_DIR ${PROJECT_BINARY_DIR}/output/python
	OUTFILE_DIR ${PROJECT_BINARY_DIR}/tmp
	SOURCES ${PROJECT_SOURCE_DIR}/python/include/math_library_interface.i ${PYTHON_DIR_LIST})

add_library(${PROJECT_NAMESPACE}::${PROJECT_PYTHON_NAME} ALIAS ${PROJECT_PYTHON_NAME})
set_property(TARGET ${PROJECT_PYTHON_NAME} PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include/)
target_include_directories(${TEST_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include/)

target_include_directories(${PROJECT_PYTHON_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/python/include ${Python3_INCLUDE_DIRS})

if (NOT("${intel_compiler}" STREQUAL "intel_compiler-NOTFOUND"))
	message("-- Using Intel Compiler")
	SET(CMAKE_CXX_COMPILER "icpc")
	SET(CMAKE_CXX_FLAGS "-Ofast ${CMAKE_CXX_FLAGS}")
	SET(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
	SET(CMAKE_CXX_FLAGS "-fopenmp ${CMAKE_CXX_FLAGS}")
	SET(CMAKE_CXX_FLAGS "-flto ${CMAKE_CXX_FLAGS}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	message("-- Using GNU Compiler")
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

target_link_libraries(${TEST_NAME} PRIVATE ${PROJECT_NAME})
target_link_libraries(${PROJECT_PYTHON_NAME} PRIVATE ${PROJECT_NAME})

set_target_properties(${PROJECT_PYTHON_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/output/python)
set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/output/test)
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/output/lib)


install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION /lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
