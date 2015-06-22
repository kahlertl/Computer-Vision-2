message(STATUS "Downloading cascades from GitHub ...")

set(CASCADES haarcascade_frontalface_alt.xml
             haarcascade_eye_tree_eyeglasses.xml)

set(GITHUB_URL "https://github.com/Itseez/opencv/raw/master/data/haarcascades")

foreach(CASCADE ${CASCADES})
    message(STATUS "${GITHUB_URL}/${CASCADE}")
    file(DOWNLOAD ${GITHUB_URL}/${CASCADE} ${DESTINATION}/${CASCADE})
endforeach()
