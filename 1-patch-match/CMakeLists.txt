add_executable(patchmatch patchmatch.cpp patchmatch.hpp main.cpp)

target_link_libraries(patchmatch wingetopt ${OpenCV_LIBS})
