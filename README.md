# Computer Vision 2

[![build status](https://square-src.de/ci/projects/7/status.png?ref=grabcut)](https://square-src.de/ci/projects/7?ref=master)

Source code for the exercises related to my Computer Vision 2 course, taken at
the TU Dresden in summer semester 2015.

The programs and scripts are written in Python using the OpenCV library. All
tests are run on an Ubuntu 14.04 LTS.

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
