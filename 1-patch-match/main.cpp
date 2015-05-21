#include <time.h>       // time
#include <stdlib.h>     // srand, rand
#include "patchmatch.hpp"
#include <iostream>
#include <getopt.h> // getopt_long()

using namespace cv;
using namespace std;

static void usage()
{
    cout << "Usage: ./stereo_match [options] left right" << endl;
    cout << "  options:" << endl;
    cout << "    -h, --help            Show this help message" << endl;
    cout << "    -r, --radius          Block radius for stereo matching. Default: 2" << endl;
    cout << "    -d, --max-disparity   Shrinks the range that will be used" << endl;
    cout << "                          for block matching. Default: 20" << endl;
    cout << "    -t, --target          Name of output file. Default: disparity.png" << endl;
    cout << "    -m, --median          Radius of the median filter applied to " << endl;
    cout << "                          the disparity map. If 0, this feature is " << endl;
    cout << "                          disabled. Default: 2" << endl;
    cout << "    -g, --ground-truth    Optimal disparity image. This activates the" << endl;
    cout << "                          search for the optimal block size for each pixel." << endl;
    cout << "                          The radius parameter will be used for the step" << endl;
    cout << "                          range [0, step]. For each element in the interval" << endl;
    cout << "                          there will be a match performed with radius = 2^step" << endl;
    cout << "                             2^step" << endl;
    cout << "    -c, --correlation     Method for computing correlation. There are:" << endl;
    cout << "                              ssd  sum of square differences" << endl;
    cout << "                              sad  sum of absolute differences" << endl;
    cout << "                              ccr  cross correlation" << endl;
    cout << "                          Default: sad" << endl;
    cout << "    -l, --lrc-threshold   Maximal distance in left-right-consistency check." <<  endl;
    cout << "                          If left and right flow must differ more than this" << endl;
    cout << "                          parameter, the region is considered as occluded." << endl;
    cout << "                          If negative, LRC will be disabled. Default: 3" << endl;
}

static bool parsePositionalImage(Mat& image, const int channels, const string& name, int argc, char const *argv[])
{
    if (optind >= argc) {
        cerr << argv[0] << ": required argument: '" << name << "'" << endl;
        usage();

        return false;
    } else {
        image = imread(argv[optind++], channels);

        if (image.empty()) {
            cerr << "Error: Cannot read '" << argv[optind] << "'" << endl;

            return false;
        }
    }

    return true;
}

int main(int argc, const char* argv[])
{
    // initialize random seed
    srand(time(nullptr));

    Mat image1;
    Mat image2;
    Mat result;
    Mat rgb;

    const struct option long_options[] = {
        { "help",           no_argument,       0, 'h' },
        { "radius",         required_argument, 0, 'r' },
        { "target",         required_argument, 0, 't' },
        { "max-disparity",  required_argument, 0, 'd' },
        { "median",         required_argument, 0, 'm' },
        { "ground-truth",   required_argument, 0, 'g' },
        { "correlation",    required_argument, 0, 'c' },
        { "lrc-threshold",  required_argument, 0, 'l' },
        0 // end of parameter list
    };

    // parse command line options
    while (true) {
        int index = -1;

        int result = getopt_long(argc, (char **) argv, "hr:t:d:m:g:c:l:", long_options, &index);

        // end of parameter list
        if (result == -1) {
            break;
        }

        switch (result) {
            case 'h':
                usage();
                return 0;

            case 'l':
                lrc_threshold = stoi(string(optarg));
                break;

            case 'r':
                radius = stoi(string(optarg));
                if (radius < 0) {
                    cerr << argv[0] << ": Invalid radius " << optarg << endl;
                    return 1;
                }
                break;

            case 'd':
                max_disparity = stoi(string(optarg));
                if (max_disparity <= 0) {
                    cerr << argv[0] << ": Invalid maximal disparity " << optarg << endl;
                    return 1;
                }
                break;

            case 't':
                target = optarg;
                break;

            case 'm':
                median_radius = stoi(string(optarg));
                if (median_radius < 0) {
                    cerr << argv[0] << ": Invalid median radius " << optarg << endl;
                    return 1;
                }
                break;

            case 'g':
                ground_truth = imread(optarg, CV_LOAD_IMAGE_GRAYSCALE);
                break;

            case 'c':
                match_name = string(optarg);

                if (match_name == "ssd") {
                    match_fn = &matchSSD;
                } else if (match_name == "sad") {
                    match_fn = &matchSAD;
                } else if (match_name == "ccr") {
                    match_fn = &matchCCR;
                } else {
                    cerr << argv[0] << ": Invalid correlation method '" << optarg << "'" << endl;
                    return 1;
                }

                break;

            case '?': // missing option
                return 1;

            default: // unknown
                cerr << "unknown parameter: " << optarg << endl;
                break;
        }
    }

    if (!parsePositionalImage(image1, CV_LOAD_IMAGE_GRAYSCALE, "frame1", argc, argv)) { return 1; }
    if (!parsePositionalImage(image2, CV_LOAD_IMAGE_GRAYSCALE, "frame2", argc, argv)) { return 1; }

    if (image1.size != image2.size) {
        cerr << "Images must be of same dimensions" << endl;
        return 1;
    }

    #ifndef NDEBUG
        cerr << "Image size: " << image1.size() << endl;
    #endif

    PatchMatch pm(200,  // maximal offsets
                    4,  // match radius
                    5); // iterations

    pm.match(image1, image2, result);

    flow2rgb(result, rgb);

    imshow("Optiocal flow", rgb);
    waitKey();

    return 0;
}
