cmake_minimum_required(VERSION 2.8)

project(cv2)

option(FORCE_WINGETOPT "help string describing option" OFF)

# Set compiler flag for C++ 2011 standard
if (CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "-std=c++0x")
endif()

# we use the OpenCV lib
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV includes ${OpenCV_INCLUDE_DIRS}")

# were does the binaries and libs are delivered?
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(ExternalProject)
ExternalProject_Add(argtable2
    PREFIX "argtable2"
    URL http://prdownloads.sourceforge.net/argtable/argtable2-13.tar.gz
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
)
ExternalProject_Get_Property(argtable2 SOURCE_DIR)
include_directories(${SOURCE_DIR}/src)

# create library we can use for linking
add_library(libargtable2 STATIC IMPORTED)
# Visual Studio adds an "d" to its debug libraries
if(MSVC AND  CMAKE_BUILD_TYPE EQUAL "DEBUG")
  set_property(TARGET libargtable2 PROPERTY IMPORTED_LOCATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libargtable2d${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set_property(TARGET libargtable2 PROPERTY IMPORTED_LOCATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libargtable2${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

add_dependencies(libargtable2 argtable2)

# file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/optionparser)
# add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/optionparser/optionparser.hpp
#                    COMMAND ${CMAKE_COMMAND} -D OUTPUT=${CMAKE_BINARY_DIR}/optionparser/optionparser.hpp
#                                             -P ${CMAKE_SOURCE_DIR}/optionparser.cmake)
# add_custom_target(optionparser DEPENDS ${CMAKE_BINARY_DIR}/optionparser/optionparser.hpp)
# set_source_files_properties(${CMAKE_BINARY_DIR}/optionparser/optionparser.hpp PROPERTIES GENERATED TRUE) 

# include_directories(${CMAKE_BINARY_DIR}/optionparser)
# # add_library(optionparser SHARED ${CMAKE_BINARY_DIR}/optionparser/optionparser.hpp)
# # set_target_properties(optionparser PROPERTIES LINKER_LANGUAGE C++)

# include the different tasks
add_subdirectory(1-patch-match/)
add_subdirectory(2-grabcut/)
