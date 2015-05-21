#include <opencv2/opencv.hpp>
#include <cstdlib>     // rand
#include <iostream>
#include <numeric> // numeric_limits
#include "patchmatch.hpp"

using namespace std;
using namespace cv;

float ssd(const Mat& image1, const Point2i& center1, const Mat& image2, const Point2i& center2,
          const int radius, const float halt)
{
    float sum = 0;

    for (int row = -radius; row <= radius; ++row) {
        for (int col = -radius; col <= radius; ++col) {
            const uchar gray1 = image1.at<uchar>(row + center1.y, col + center1.x);
            const uchar gray2 = image2.at<uchar>(row + center2.y, col + center2.x);

            const float diff  = gray1 - gray2;

            sum += diff * diff;

            // early termination
            if (sum > halt) {
                return sum;
            }
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



PatchMatch::PatchMatch(int maxoffset, int match_radius, int iterations, int pyramid,
                       float search_ratio, int search_radius) :
    // Parameters
    iterations(iterations),
    pyramid(pyramid),
    maxoffset(maxoffset),
    match_radius(match_radius),
    search_ratio(search_ratio),
    border(match_radius),
    max_search_radius(search_radius == -1),
    search_radius(search_radius)
{
    // do nothing
}

void PatchMatch::match(const Mat& image1, const Mat& image2, Mat& dest)
{
    vector<tuple<Mat, Mat>> levels(pyramid);
    levels[0] = tuple<Mat, Mat>(image1, image2);

    for (int p = 1; p < pyramid; ++p) {
        tuple<Mat, Mat> level;

        resize(get<0>(levels[p - 1]), get<0>(level), Size(), 2.0 / 3.0, 2.0 / 3.0);
        resize(get<1>(levels[p - 1]), get<1>(level), Size(), 2.0 / 3.0, 2.0 / 3.0);

         // imshow("level", get<0>(level));
         // waitKey();

        cerr << "new level " << p << endl;

        levels[p] = level;
    }

    // walk backwards through the pyramid levels
    for (int p = pyramid - 1; p >= 0; --p) {
        #ifndef NDEBUG
            cerr << "Pyramid level " << p << endl;
        #endif

        Mat frame1 = get<0>(levels[p]);
        Mat frame2 = get<1>(levels[p]);

        imshow("frame 1", frame1);
        waitKey();

        nrows = frame1.rows;
        ncols = frame2.cols;

        if (max_search_radius == true) {
            search_radius = min(nrows, ncols);
        }

        // the very first iteration, we have to initialize the offsets randomly
        if (p == pyramid - 1) {
            // create an empty matrix with the same x-y dimensions like the first
            // image but with two channels. Each channel stands for an x/y offset
            // of a pixel at this position.
            flow = Mat::zeros(nrows, ncols, CV_32FC2); // 2-channel 32-bit floating point

            cerr << "initialize" << endl;
            initialize(frame1, frame2);
        }
        // in the lower pyramid levels, we can use the prior knowledge and scale the
        // offset matrix up
        else {
            Mat resized;
            resize(flow, resized, frame1.size());

            flow = resized;

            Mat rgb;
            flow2rgb(flow, rgb);
            imshow("flow", rgb);
            waitKey();
        }

        for (niterations = 0; niterations < iterations; ++niterations) {

            #ifndef NDEBUG
                cerr << "iteration " << (niterations + 1) << endl;
            #endif

            // for (int row = match_radius; row < nrows - match_radius; ++row) {
            for (int row = border; row < nrows - border ; ++row) {

                #ifndef NDEBUG
                    cerr << "\r" << row;
                #endif

                for (int col = border; col < ncols - border; ++col) {
                    float cost = propagate(frame1, frame2, row, col);
                    random_search(frame1, frame2, row, col, cost);
                }
            }
            #ifndef NDEBUG
                cerr << "\r";
            #endif
        }

        Mat rgb;
        flow2rgb(flow, rgb);
        imshow("flow", rgb);
        waitKey();
    }


    flow.copyTo(dest);
}

void PatchMatch::initialize(const Mat& image1, const Mat& image2)
{
    Point2i offset;
    Point2i index;
    Point2i pixel;

    for (int row = border; row < nrows - border ; ++row) {
        for (int col = border; col < ncols - border ; ++col) {
            index.x = col;
            index.y = row;

            // search for an offset that leads to a pixel inside the other image
            while (true) {
                offset.x = rand() % (2 * maxoffset) - maxoffset;
                offset.y = rand() % (2 * maxoffset) - maxoffset;

                pixel = index + offset;

                // check if the pixel is inside the other image
                if (in_borders(pixel)) {
                    break;
                }
            }
            flow.at<Point2f>(row, col) = offset;
        }
    }
}

float PatchMatch::propagate(const cv::Mat &image1, const cv::Mat &image2, const int row, const int col)
{
    // switch between top and left neighbor in even iterations and
    // right bottom neighbor in odd iterations
    int direction = (niterations % 2 == 0) ? 1 : -1;

    Point2f index(col, row);

    Point2f pixel      = index + flow.at<Point2f>(row, col);
    Point2f y_neighbor = index + flow.at<Point2f>(row + direction, col);  // top or bottom neighbor
    Point2f x_neighbor = index + flow.at<Point2f>(row, col + direction);  // left or right neighbor

    // Point2f indices[3] = {
    //     flow.at<Point2f>(row, col),
    //     flow.at<Point2f>(row + direction, col),
    //     flow.at<Point2f>(row , col + direction)
    // };
    // float costs[3];

    // for (int i = 0; i < 3; ++i) {
    //     Point2f center = index + indices[i];

    //     if (in_borders(center)) {
    //         costs[i] = ssd(image1, index, image2, center, match_radius);
    //     } else {
    //         costs[i] = numeric_limits<float>::infinity();
    //     }
    // }

    // int minindex = 0;

    // for (int i = 1; i < 3; ++i) {
    //     if (costs[i] < costs[minindex]) {
    //         minindex = i;
    //     }
    // }

    // flow.at<Point2f>(row, col) = indices[minindex];

    float costs = ssd(image1, index, image2, pixel, match_radius);

    // x-direction (left or right)
    if (in_borders(x_neighbor)) {
        float x_costs = ssd(image1, index, image2, x_neighbor, match_radius);

        // update offset if the costs of offset of the neighbor in y-direction
        // is smaller
        if (x_costs < costs) {
            costs = x_costs;
            flow.at<Point2f>(row, col) = flow.at<Point2f>(row, col + direction);
        }
    }

    // y-direction (top or bottom)
    if (in_borders(y_neighbor)) {
        float y_costs = ssd(image1, index, image2, y_neighbor, match_radius);

        // update offset if the costs of offset of the neighbor in y-direction
        // is smaller
        if (y_costs < costs) {
             costs = y_costs;
            flow.at<Point2f>(row, col) = flow.at<Point2f>(row + direction, col);
        }
    }

    return costs;
}


void PatchMatch::random_search(const cv::Mat &image1, const cv::Mat &image2, const int row, const int col, float costs)
{
    const Point2f index(col, row);
    int i = 0;

    while (true) {
        const float distance = search_radius * pow(search_ratio, i++);

        // halt condition. search radius must not be smaller
        // than one pixel
        if (distance < 1) {
            break;
        }

        // jump randomly in the interval [-1, 1] x [-1, 1]
        Point2f offset = random_interval();
        offset.x *= distance;
        offset.y *= distance;

        // calculate center of pixel in the other image
        Point2i center = offset + index;

        // check if the selected offset is valid, that means:
        //  - it has to be inside the max offset bound
        //  - must be inside the image
        if (abs(offset.x) <= maxoffset && abs(offset.y) <= maxoffset &&  in_borders(center)) {
            float match = ssd(image1, Point2i(row, col), image2, center, search_radius, costs);

            // if better match was found, update the current costs and insert the offset
            if (match < costs) {
                costs = match;
                flow.at<Point2f>(index) = offset;
            }
        }
    }
}

bool PatchMatch::in_borders(Point2i point) {
    return border <= point.x && point.x < ncols - border &&
           border <= point.y && point.y < nrows - border;
}
