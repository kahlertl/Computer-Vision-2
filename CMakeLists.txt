cmake_minimum_required(VERSION 2.8)

project(cv2)

option(FORCE_WINGETOPT "help string describing option" OFF)

# Set compiler flag for C++ 2011 standard
if (CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "-std=c++0x")
endif()

# Force usage of the Windows getopt implementation
if (FORCE_WINGETOPT)
  add_definitions("-DFORCE_WINGETOPT")
endif()

# we use the OpenCV lib
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV includes ${OpenCV_INCLUDE_DIRS}")

# were does the binaries and libs are delivered?
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Windows port of POSIX "getopt.h"
include_directories(wingetopt/)
add_subdirectory(wingetopt/)

# include the different tasks
add_subdirectory(1-patch-match/)
add_subdirectory(2-grabcut/)
