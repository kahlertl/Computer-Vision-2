add_executable(patchmatch patchmatch.cpp patchmatch.hpp main.cpp)

target_link_libraries(patchmatch ${OpenCV_LIBS})
