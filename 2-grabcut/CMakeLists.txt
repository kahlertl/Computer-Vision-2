add_executable(grabcut app.cpp grabcut.cpp)
add_executable(faceseg faceseg.cpp grabcut.cpp)

target_link_libraries(grabcut getopt ${OpenCV_LIBS})
target_link_libraries(faceseg getopt ${OpenCV_LIBS})

set(images images/face1.png
           images/face2.png
           images/face3.png
           images/face4.png
           images/lama.bmp
           images/fruits.png)

set(cascades haarcascade_frontalface_alt.xml
             haarcascade_profileface.xml
             haarcascade_eye.xml
             haarcascade_eye_tree_eyeglasses.xml)

# Tell CMake how to build the images and cascades
add_custom_command(OUTPUT ${images}
                   COMMAND ${CMAKE_COMMAND} -D SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/images"
                                            -D DESTINATION="${CMAKE_BINARY_DIR}/images"
                                            -P ${CMAKE_CURRENT_SOURCE_DIR}/images.cmake)

add_custom_command(OUTPUT ${cascades}
                   COMMAND ${CMAKE_COMMAND} -D SOURCE="${CMAKE_CURRENT_SOURCE_DIR}"
                                            -D DESTINATION="${CMAKE_BINARY_DIR}"
                                            -P ${CMAKE_CURRENT_SOURCE_DIR}/cascades.cmake)

add_custom_target(grabcut-images DEPENDS ${images})
add_custom_target(haar-cascades  DEPENDS ${cascades})
