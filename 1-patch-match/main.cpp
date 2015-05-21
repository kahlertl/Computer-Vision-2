#include <time.h>       // time
#include <stdlib.h>     // srand, rand
#include "patchmatch.hpp"
#include <iostream>
#include <getopt.h> // getopt_long()

using namespace cv;
using namespace std;

// parameters
static int match_radius  =  4;
static int maxoffset     = 20;
static int search_radius = -1;
static int iterations    =  4;
static int pyramid       =  3;

// command line option list
static const struct option long_options[] = {
    { "help",           no_argument,       0, 'h' },
    { "maxoffset",      required_argument, 0, 'm' },
    { "search-radius",  required_argument, 0, 's' },
    { "iterations",     required_argument, 0, 'i' },
    { "pyramid",        required_argument, 0, 'p' },
    { "match-radius",   required_argument, 0, 'r' },
    0 // end of parameter list
};

static void usage()
{
    cout << "Usage: patchmatch [options] image1 image2" << endl;
    cout << "  options:" << endl;
    cout << "    -h, --help            Show this help message" << endl;
    cout << "    -m, --maxoffset       Maximal offset in x and y direction for each" << endl;
    cout << "                          pixel. Default: " << maxoffset << endl;
    cout << "    -r, --radius          Block radius for template matching." << endl;
    cout << "                          Default: " << match_radius  << endl;
    cout << "    -s, --search-radius   Block radius for the random search window." << endl;
    cout << "                          If -1, the whole image will be searched." << endl;
    cout << "                          Default: " << search_radius << endl;
    cout << "    -i, --iterations      Number of iterations. Default: " << iterations << endl;
    cout << "    -p, --pyramid         Number of pyramid levels. Default: " << pyramid << endl;

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
    Mat flow;
    Mat rgb;


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

            case 'm':
                maxoffset = stoi(string(optarg));
                if (maxoffset < 0) {
                    cerr << argv[0] << ": Invalid maximal offset " << optarg << endl;
                    return 1;
                }
                break;

            case 's':
                search_radius = stoi(string(optarg));
                if (search_radius < 0) {
                    cerr << argv[0] << ": Invalid maximal offset " << optarg << endl;
                    return 1;
                }
                break;

            case 'i':
                iterations = stoi(string(optarg));
                if (iterations < 0) {
                    cerr << argv[0] << ": Invalid iterations number " << optarg << endl;
                    return 1;
                }
                break;

            case 'p':
                pyramid = stoi(string(optarg));
                if (pyramid < 0) {
                    cerr << argv[0] << ": Invalid pyramid levels " << optarg << endl;
                    return 1;
                }
                break;

            case 'r':
                match_radius = stoi(string(optarg));
                if (match_radius < 0) {
                    cerr << argv[0] << ": Invalid match radius " << optarg << endl;
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

    pm.match(image1, image2, flow);

    flow2rgb(flow, rgb);

    imshow("Optiocal flow", rgb);
    waitKey();

    return 0;
}
