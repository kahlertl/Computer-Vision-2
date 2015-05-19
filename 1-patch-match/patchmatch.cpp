#include <opencv2/opencv.hpp>
#include <cstdlib>     // rand
#include <iostream>
#include "patchmatch.hpp"

using namespace std;
using namespace cv;

float ssd(const Mat& image1, const Point2i center1, const Mat& image2, const Point2i center2, const int radius)
{
    float sum = 0;

    for (int row = -radius; row <= radius; ++row) {
        for (int col = -radius; col <= radius; ++col) {
            const uchar gray1 = image1.at<uchar>(row + center1.y, col + center1.x);
            const uchar gray2 = image2.at<uchar>(row + center2.y, col + center2.x);

            const float diff  = gray1 - gray2;

            sum += diff * diff;
        }
    }

    return sum;
}


void flow2rgb(const Mat& flow, Mat& rgb)
{
    // extract x and y channels
    Mat xy[2]; // x,y

    split(flow, xy);

    // calculate angle and magnitude for the HSV color wheel
    Mat magnitude;
    Mat angle;

    cartToPolar(xy[0], xy[1], magnitude, angle, true);

    // translate magnitude to range [0,1]
    double mag_max;
    minMaxLoc(magnitude, 0, &mag_max);

    magnitude.convertTo(
        magnitude,    // output matrix
        -1,           // type of the ouput matrix, if negative same type as input matrix
        1.0 / mag_max // scaling factor
    );

    // build HSV image (hue-saturation-value)
    Mat _hsv[3]; // array of three matrices - one for each channel
    Mat hsv;

    // create separate channels
    _hsv[0] = angle;                           // H (hue)              [0,360]
    _hsv[1] = magnitude;                       // S (saturation)       [0,1]
    _hsv[2] = Mat::ones(angle.size(), CV_32F); // V (brigthness value) [0,1]

    // merge the three components to a three channel HSV image
    merge(_hsv, 3, hsv);

    // convert to BGR
    cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
}



PatchMatch::PatchMatch(int maxoffset, int match_radius, int iterations, float search_ratio, int search_radius) :
    // Parameters
    iterations(iterations),
    maxoffset(maxoffset),
    match_radius(match_radius),
    search_ratio(search_ratio),
    max_search_radius(search_radius == -1)
{
    this->search_radius = search_radius;
}

void PatchMatch::match(const Mat& image1, const Mat& image2, Mat& dest)
{
    nrows = image1.rows;
    ncols = image1.cols;

    if (max_search_radius == -1) {
        this->search_radius = min(nrows, ncols);
    }

    border = match_radius + maxoffset;

    // create an empty matrix with the same x-y dimensions like the first
    // image but with two channels. Each channel stands for an x/y offset
    // of a pixel at this position.
    quality = Mat::zeros(nrows, ncols, CV_32FC1);
    result = Mat::zeros(nrows, ncols, CV_32FC2); // 2-channel 32-bit floating point

    initialize(image1, image2);

    for (niterations = 0; niterations < iterations; ++niterations) {
        propagate(image1, image2);
        random_search(image1, image2);
    }

    result.copyTo(dest);
}

void PatchMatch::initialize(const Mat& image1, const Mat& image2)
{
    Point2i offset;
    Point2i index;

    for (int row = border; row < nrows - border ; ++row) {
        for (int col = border; col < ncols - border ; ++col) {
            index.x = row;
            index.y = col;

            offset.x = rand() % (2 * maxoffset) - maxoffset;
            offset.y = rand() % (2 * maxoffset) - maxoffset;

            result.at<Point2f>(row, col) = offset;
            quality.at<float>(row, col) = ssd(image1, index, image2, index + offset, match_radius);
        }
    }
}

void PatchMatch::propagate(const cv::Mat &image1, const cv::Mat &image2)
{
    // TODO: implement propagation
}

void PatchMatch::random_search(const cv::Mat &image1, const cv::Mat &image2)
{
    // TODO: implement random search
}
