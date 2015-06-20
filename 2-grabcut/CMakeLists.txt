add_executable(grabcut app.cpp grabcut.cpp)

target_link_libraries(grabcut ${OpenCV_LIBS})

set(images images/face4.png
           images/lama.bmp
           images/fruits.png)

add_custom_command(OUTPUT ${images}
                   COMMAND ${CMAKE_COMMAND} -D SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/images"
                                            -D DESTINATION="${CMAKE_BINARY_DIR}/images"
                                            -P ${CMAKE_CURRENT_SOURCE_DIR}/images.cmake)

add_custom_target(grabcut-images DEPENDS ${images})
