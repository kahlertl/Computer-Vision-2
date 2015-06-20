# 
# Download and copy test images for the GrabCut exercise
# 
message(STATUS "Downloading fruits.jpg")
file(DOWNLOAD https://raw.githubusercontent.com/Itseez/opencv/master/samples/data/fruits.jpg ${DESTINATION}/fruits.jpg)

file(COPY ${SOURCE}/lama.bmp ${SOURCE}/face4.png
     DESTINATION ${DESTINATION})
