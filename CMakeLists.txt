cmake_minimum_required(VERSION 2.8)

project(cv2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

# we use the OpenCV lib
find_package(OpenCV REQUIRED)

# were does the binaries and libs are delivered?
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


if(WIN32)
  include(ExternalProject)
  ExternalProject_Add(
    wingetopt
    GIT_REPOSITORY https://github.com/alex85k/wingetopt.git
    PREFIX wingetopt
    # GIT_REPOSITORY https://square-src.de/gitlab/lucas/computer-vision-2.git
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    LOG_DOWNLOAD ON
    LOG_INSTALL ON
  )
  include_directories(${CMAKE_BINARY_DIR}/include/)
endif()



# include the different tasks
add_subdirectory(1-patch-match/)
add_subdirectory(2-grabcut/)
