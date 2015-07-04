# Computer Vision 2

[![build status](https://square-src.de/ci/projects/7/status.png?ref=grabcut)](https://square-src.de/ci/projects/7?ref=master)

Source code for the exercises related to my Computer Vision 2 course, taken at
the TU Dresden in summer semester 2015.

The programs and scripts are written in Python using the OpenCV library. All
tests are run on an Ubuntu 14.04 LTS.

## Building on Windows

First, choose a Windows C++ compiler. I have chosen [MinGW64](http://mingw-w64.org/doku.php))
because the normal MinGW has a [bug with C++11](http://stackoverflow.com/q/8542221/2467158)
- and I like software that is up-to-date.

### Installation of MinGW64

Because I have got an error with the Windows installer of MinGW64, so I decided to download
the build directly from [SourceForge.net][1].

### Building OpenCV

Because the normal OpenCV Windows distribution does not contain a build for
MinGW64, we have to build it on our own.

First, download [OpenCV package][2] and run the installer. Open a command line
(`Win+R` and `cmd`) and navigate to the location where you extracted the OpenCV lib.

```batch
% create a directory for the new build
mkdir mingw64build
cd mingw64build

% configure with cmake - this can also be done with CMake-GUI
% If you have no OpenCL, pass "-D WITH_OPENCL=OFF" as additional definition
cmake -G "MinGW Makefiles" -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++  -D CMAKE_C_COMPILER=gcc  ..\sources

% build
mingw32-make opencv_modules
```

After this, you should create a new environmental variable called "OpenCV_DIR"
which targets the `mingw64build` directory. If you do this, you do not need to
configure the OpenCV location for each separate project.

### Building tasks

```batch
% create build directory
mkdir build
cd build

% building
cmake -G "MinGW Makefiles" -D CMAKE_BUILD_TYPE=Debug -D CMAKE_CXX_COMPILER=g++ MAKE_MAKE_PROGRAM=mingw32-make ..
```

## Python implementations

## Requirements

```bash
# install OpenCV as python package together with python-numpy
$ sudo apt-get install python-opencv
```

## C++ implementations

The C++ implementation uses cmake as build system

```bash

$ mkdir build
$ cd build/
# You can also specify "Debug" as build type to get some more verbose
# print statements
$ cmake -D CMAKE_BUILD_TYPE=Release ..
$ make patchmatch

# run your binary
$ bin/patchmatch ../frame1.png ../frame2.png
```

[1]: http://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/5.1.0/threads-posix/seh/x86_64-5.1.0-release-posix-seh-rt_v4-rev0.7z/download
[2]: http://sourceforge.net/projects/opencvlibrary/files/opencv-win/
