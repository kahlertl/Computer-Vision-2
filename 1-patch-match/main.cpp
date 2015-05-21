#include <time.h>       // time
#include <stdlib.h>     // srand, rand
#include "patchmatch.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, const char* argv[])
{
    // initialize random seed
    srand(time(nullptr));

    Mat image1;
    Mat image2;
    Mat result;
    Mat rgb;

    if (argc != 3) {
        cerr << "Usage: patchmatch <image1> <image2>" << endl;
        return 1;
    }

    image1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    image2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

    if (image1.empty()) {
        cerr << "Cannot read image 1" << endl;
        return 1;
    }
    if (image2.empty()) {
        cerr << "Cannot read image 2" << endl;
        return 1;
    }

    PatchMatch pm(10, 3);

    pm.match(image1, image2, result);

    flow2rgb(result, rgb);

    imshow("Optiocal flow", rgb);
    waitKey();

    return 0;
}
