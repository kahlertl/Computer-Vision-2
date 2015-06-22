# 
# Download and copy test images for the GrabCut exercise
# 
message(STATUS "Downloading fruits.jpg")
file(DOWNLOAD https://raw.githubusercontent.com/Itseez/opencv/master/samples/data/fruits.jpg ${DESTINATION}/fruits.jpg)

set(IMAGES lama.bmp
           face1.png
           face2.png
           face3.png
           face4.png)

foreach(IMAGE ${IMAGES})
    file(COPY ${SOURCE}/${IMAGE}
         DESTINATION ${DESTINATION})
endforeach()

