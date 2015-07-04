#include <iostream>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "wingetopt.h" // POSIX getopt_long()
#include "grabcut.hpp"

using namespace std;
using namespace cv;

// command line option list
static const struct option long_options[] = {
    { "help",       no_argument,       0, 'h' },
    { "extended",   no_argument,       0, 'e' },
    { "neighbors",  required_argument, 0, 'n' },
    0 // end of parameter list
};

static void usage()
{
    cout << "Usage: grabcut [options] image" << endl
         << endl
         << "This program demonstrates GrabCut segmentation." << endl
         << "Select an object in a region and then grabcut will attempt to segment it out." << endl
         << endl
         << "  options:" << endl
         << "    -h, --help            Show this help message" << endl
         << "    -e, --extended        Use an extended pairwise (aka. binary or smoothing) term" << endl
         << "                          Default: false" << endl
         << "    -n, --neighbors       Neighborhood system that should be used." << endl
         << "                          Default: 8" << endl
         << endl;
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

static void hotkeyHelp()
{
    cout << endl
         << "Select a rectangular area around the object you want to segment" << endl
         << endl
         << "Hot keys: " << endl
         << "    ESC - quit the program" << endl
         << "    r   - restore the original image" << endl
         << "    n   - next iteration" << endl
         << endl
         << "    left mouse button - set rectangle" << endl
         << endl
         << "    CTRL +left mouse button - set GC_BGD pixels" << endl
         << "    SHIFT+left mouse button - set CG_FGD pixels" << endl
         << endl
         << "    CTRL +right mouse button - set GC_PR_BGD pixels" << endl
         << "    SHIFT+right mouse button - set CG_PR_FGD pixels" << endl
         << endl;
}

// color definitions
const Scalar RED       = Scalar(  0,   0, 255);
const Scalar PINK      = Scalar(230, 130, 255);
const Scalar BLUE      = Scalar(255,   0,   0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN     = Scalar(  0, 255,   0);

const int BGD_KEY = CV_EVENT_FLAG_CTRLKEY;
const int FGD_KEY = CV_EVENT_FLAG_SHIFTKEY;

const int MAX_TOLERANCE     =   100;
const int MAX_CONNECTIVITY  = 10000;
const int MAX_CONTRAST      = 10000;

static void getBinMask(const Mat &comMask, Mat &binMask)
{
    if (comMask.empty() || comMask.type() != CV_8UC1) {
        CV_Error(CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
    }
    if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols) {
        binMask.create(comMask.size(), CV_8UC1);
    }
    binMask = comMask & 1;
}

class GCApplication
{
  public:
    // state flags
    enum
    {
        NOT_SET = 0, IN_PROCESS = 1, SET = 2
    };

    // parameters for brush stroke painting
    static const int radius = 2;
    static const int thickness = -1;

    GCApplication(double tolerance = MAX_TOLERANCE, double connectivity = 1, double contrast = 1,
                  bool extended = false, int neighbors = GC_N8) :
        tolerance(tolerance),
        connectivity(connectivity),
        contrast(contrast),
        extended(extended),
        neighbors(neighbors),
        image(nullptr),
        winName(nullptr)
    {}

    void reset();

    void setImageAndWinName(const Mat &_image, const string &_winName);

    void showImage() const;

    void mouseClick(int event, int x, int y, int flags);

    int nextIter();

    /**
     * Sets tolerance for initial GMM estimization. The app will fallback
     * in an uninitiazed state with, but the rectangular and brush strokes
     * are kept.
     */
    inline void setTolerance(double _tolerance)       { tolerance = _tolerance;        resetIter(); }
    inline void setContrast(double _contrast)         { contrast  = _contrast;         resetIter(); }
    inline void setConnectivity(double _connectivity) { connectivity  = _connectivity; resetIter(); }

    inline int getIterCount() const { return iterCount; }

    friend ostream& operator<< (ostream& stream, const GCApplication& gcapp);
  private:
    void setRectInMask();

    void setLabelsInMask(int flags, Point p, bool isPr);

    void resetIter();

    const string *winName;
    const Mat *image;
    Mat mask;

    // temporary arrays fir the back- and foreground model. They will be used internally by GrabCut
    // and should not be modified during the interations
    Mat backgroundModel, foregroundModel;

    // state flags
    uchar rectState, labelsState, probablyLabelsState;
    bool isInitialized;

    // region of interest containing a segmented object. defined by user
    Rect rect;
    // vectors of user-marked pixels for fore- and background
    vector<Point> foregroundPixels, backgroundPixels, probablyForegroundPixels, probablyBackgroundPixels;
    int iterCount;

    double tolerance;       // Defines the proportion of pixels inside the rectangular that
                            // are most unlikly in the background distribution and should be used
                            // for the foreground distribution
    double connectivity;    // standard Ising prior, which are the constant costs for the
                            // pairwise / binary / smoothing term (delta function with proportinal factor)
    double contrast;        // boost cutting of edges in hight contrast reagions
    bool extended;          // use an extended term of the pairwise term
    int neighbors;          // numer of connected neighbor pixels (aka. graph connectivity)
};

void GCApplication::reset()
{
    // reset the mask to all background
    if (!mask.empty()) {
        mask.setTo(Scalar::all(GC_BGD));
    }
    backgroundPixels.clear();
    foregroundPixels.clear();
    probablyBackgroundPixels.clear();
    probablyForegroundPixels.clear();

    isInitialized = false;
    rectState = NOT_SET;
    labelsState = NOT_SET;
    probablyLabelsState = NOT_SET;
    iterCount = 0;
}

void GCApplication::setImageAndWinName(const Mat &_image, const string &_winName)
{
    if (_image.empty() || _winName.empty()) {
        return;
    }
    image = &_image;
    winName = &_winName;
    mask.create(image->size(), CV_8UC1);
    reset();
}

void GCApplication::showImage() const
{
    if (image == nullptr || winName == nullptr || image->empty() || winName->empty()) {
        return;
    }

    if (isInitialized) {
        Mat binMask;
        // create initial binary mask
        getBinMask(mask, binMask);
        
        Mat segmentation;
        image->copyTo(segmentation, binMask);
        
        namedWindow("segmentation", WINDOW_AUTOSIZE);
        imshow("segmentation", segmentation);
    }

    Mat canvas;
    image->copyTo(canvas);

    // draw each user-defined brush stroke
    for (int i = 0; i < backgroundPixels.size(); ++i) {
        circle(canvas, backgroundPixels[i], radius, BLUE, thickness);
    }
    for (int i = 0; i < foregroundPixels.size(); ++i) {
        circle(canvas, foregroundPixels[i], radius, RED, thickness);
    }
    for (int i = 0; i < probablyBackgroundPixels.size(); ++i) {
        circle(canvas, probablyBackgroundPixels[i], radius, LIGHTBLUE, thickness);
    }
    for (int i = 0; i < probablyForegroundPixels.size(); ++i) {
        circle(canvas, probablyForegroundPixels[i], radius, PINK, thickness);
    }

    if (rectState == IN_PROCESS || rectState == SET) {
        rectangle(canvas, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), GREEN, 2);
    }

    imshow(*winName, canvas);
}

void GCApplication::setRectInMask()
{
    assert(!mask.empty());
    mask.setTo(GC_BGD);
    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, image->cols - rect.x);
    rect.height = min(rect.height, image->rows - rect.y);
    (mask(rect)).setTo(Scalar(GC_PR_FGD));
}

void GCApplication::setLabelsInMask(int flags, Point p, bool isPr)
{
    vector<Point> *bpxls, *fpxls;
    uchar bvalue, fvalue;
    if (!isPr) {
        bpxls = &backgroundPixels;
        fpxls = &foregroundPixels;
        bvalue = GC_BGD;
        fvalue = GC_FGD;
    } else {
        bpxls = &probablyBackgroundPixels;
        fpxls = &probablyForegroundPixels;
        bvalue = GC_PR_BGD;
        fvalue = GC_PR_FGD;
    }
    if (flags & BGD_KEY) {
        bpxls->push_back(p);
        circle(mask, p, radius, bvalue, thickness);
    }
    if (flags & FGD_KEY) {
        fpxls->push_back(p);
        circle(mask, p, radius, fvalue, thickness);
    }
}

void GCApplication::mouseClick(int event, int x, int y, int flags)
{
    // TODO add bad args check
    switch (event) {
        case CV_EVENT_LBUTTONDOWN: { // set rect or GC_BGD(GC_FGD) labels
            bool isb = (flags & BGD_KEY) != 0,
                isf = (flags & FGD_KEY) != 0;

            if (rectState == NOT_SET && !isb && !isf) {
                rectState = IN_PROCESS;
                rect = Rect(x, y, 1, 1);
            }
            if ((isb || isf) && rectState == SET) {
                labelsState = IN_PROCESS;
            }
        }
            break;
        case CV_EVENT_RBUTTONDOWN: { // set GC_PR_BGD(GC_PR_FGD) labels
            bool isb = (flags & BGD_KEY) != 0,
                isf = (flags & FGD_KEY) != 0;
            if ((isb || isf) && rectState == SET) {
                probablyLabelsState = IN_PROCESS;
            }
        }
            break;
        case CV_EVENT_LBUTTONUP:
            if (rectState == IN_PROCESS) {
                rect = Rect(Point(rect.x, rect.y), Point(x, y));
                rectState = SET;
                setRectInMask();
                assert(backgroundPixels.empty() && foregroundPixels.empty() && probablyBackgroundPixels.empty() &&
                       probablyForegroundPixels.empty());
                showImage();
            }
            if (labelsState == IN_PROCESS) {
                setLabelsInMask(flags, Point(x, y), false);
                labelsState = SET;
                showImage();
            }
            break;
        case CV_EVENT_RBUTTONUP:
            if (probablyLabelsState == IN_PROCESS) {
                setLabelsInMask(flags, Point(x, y), true);
                probablyLabelsState = SET;
                showImage();
            }
            break;
        case CV_EVENT_MOUSEMOVE:
            if (rectState == IN_PROCESS) {
                rect = Rect(Point(rect.x, rect.y), Point(x, y));
                assert(backgroundPixels.empty() && foregroundPixels.empty() && probablyBackgroundPixels.empty() &&
                       probablyForegroundPixels.empty());
                showImage();
            } else if (labelsState == IN_PROCESS) {
                setLabelsInMask(flags, Point(x, y), false);
                showImage();
            } else if (probablyLabelsState == IN_PROCESS) {
                setLabelsInMask(flags, Point(x, y), true);
                showImage();
            }
            break;
        default:
            break;
    }
}

int GCApplication::nextIter()
{
    if (isInitialized) {
        extendedGrabCut(*image, mask, rect, backgroundModel, foregroundModel, 1,
                        tolerance, extended, connectivity, contrast, neighbors);
    } else {
        // if the application not initialized and the rectangular is not set up be the user
        // we do nothing
        if (rectState != SET) {
            return iterCount;
        }

        // do the initial iteration
        //
        // if the user provides brush strokes, use them as mask for the initial iteration
        if (labelsState == SET || probablyLabelsState == SET) {
            extendedGrabCut(*image, mask, rect, backgroundModel, foregroundModel, 1,
                            tolerance, extended, connectivity, contrast, neighbors, GC_INIT_WITH_MASK);
        } else {
            extendedGrabCut(*image, mask, rect, backgroundModel, foregroundModel, 1,
                            tolerance, extended, connectivity, contrast, neighbors, GC_INIT_WITH_RECT);
        }
        // after the initial iteration, the application is initialized
        isInitialized = true;
    }
    iterCount++;

    // delete all brush strokes
    // backgroundPixels.clear();
    // foregroundPixels.clear();
    // probablyBackgroundPixels.clear();
    // probablyForegroundPixels.clear();

    return iterCount;
}

void GCApplication::resetIter()
{
    isInitialized = false;
    iterCount = 0;
    showImage();
}

ostream& operator<< (ostream& stream, const GCApplication& gcapp)
{
    return stream << "GCApplication \"" << gcapp.winName << "\"" << endl
                  << "    extended binary:  " << (gcapp.extended ? "true" : "false") << endl
                  << "    neighbors:        " << gcapp.neighbors << endl
                  << "    tolerance:        " << gcapp.tolerance << endl
                  << "    connectivity:     " << gcapp.connectivity  << endl
                  << "    contrast:         " << gcapp.contrast  << endl;
}

static inline double trackbarToTolerance(int value)    { return (double)  value / (double) MAX_TOLERANCE; }
static inline double trackbarToContrast(int value)     { return (double)  100 * value / (double) MAX_CONTRAST;  }
static inline double trackbarToConnectivity(int value) { return (double)  100 * value / (double) MAX_CONNECTIVITY;  }

static void onMouse(int event, int x, int y, int flags, void* gcapp)
{
    ((GCApplication*) gcapp)->mouseClick(event, x, y, flags);
}

static void onToleranceTrackbar(int value, void* gcapp)
{
    ((GCApplication*) gcapp)->setTolerance(trackbarToTolerance(value));
}

static void onContrastTrackbar(int value, void* gcapp)
{
    ((GCApplication*) gcapp)->setContrast(trackbarToContrast(value));
}

static void onConnectivityTrackbar(int value, void* gcapp)
{
    ((GCApplication*) gcapp)->setConnectivity(trackbarToConnectivity(value));
}

int main(int argc, const char **argv)
{
    Mat image;

    // command line parameters
    int neighbors = GC_N8;
    bool extended = false;

    // trackbar parameters
    int toleranceSlider = 50;
    int connectivitySlider  = 100;
    int contrastSlider  = 100;

    // parse command line options
    while (true) {
        int index = -1;
        int result = getopt_long(argc, (char **) argv, "hen:", long_options, &index);

        // end of parameter list
        if (result == -1) {
            break;
        }

        switch (result) {
            case 'h':
                usage();
                return 0;

            case 'e':
                extended = true;
                break;

            case 'n': {
                int n = stoi(string(optarg));
                if (n == 4) {
                    neighbors = GC_N4;
                } else if (n == 8) {
                    neighbors = GC_N8;
                } else {
                    cerr << argv[0] << ": Invalid neighborshood " << optarg
                         << ". Got to town, only 4 and 8 are supported. Implement it yourself!" << endl;
                    return 1;
                }
                break;
            }

            case '?': // missing option
                return 1;

            default: // unknown
                cerr << "unknown parameter: " << optarg << endl;
                break;
        }
    }

    // load remaining command line argument
    if (!parsePositionalImage(image, CV_LOAD_IMAGE_COLOR, "image", argc, argv)) { return 1; }

    // use the inital value of the slider for the app initialziation
    GCApplication gcapp(trackbarToTolerance(toleranceSlider),
                        trackbarToConnectivity(connectivitySlider),
                        trackbarToContrast(contrastSlider),
                        extended,
                        neighbors);

    const string winName = "image";
    namedWindow(winName, WINDOW_AUTOSIZE);

    setMouseCallback(winName, onMouse, &gcapp);

    createTrackbar("tolerance",    winName, &toleranceSlider,    MAX_TOLERANCE,    onToleranceTrackbar,    &gcapp);
    // we do not need the extra trackbars if the option is not set
    if (extended) {
        createTrackbar("connectivity", winName, &connectivitySlider, MAX_CONNECTIVITY, onConnectivityTrackbar, &gcapp);
        createTrackbar("constrast",    winName, &contrastSlider,     MAX_CONTRAST,     onContrastTrackbar,     &gcapp);
    }

    //
    gcapp.setImageAndWinName(image, winName);

    // print settings and help
    cout << gcapp << endl;
    hotkeyHelp();

    gcapp.showImage();

    while (true) {
        int c = waitKey(0);
        switch ((char) c) {
            case '\x1b':
                cout << "Exiting ..." << endl;
                goto exit_main;
            case 'r':
                cout << endl;
                gcapp.reset();
                gcapp.showImage();
                break;
            case 'n': {
                // we need the curly brackets for scope reasons. Otherwise we
                // the compiler cries, because we could skip the initialziation
                // of the iterCount   variable
                int iterCount = gcapp.getIterCount();
                cout << "<" << iterCount << "... ";
                int newIterCount = gcapp.nextIter();
                if (newIterCount > iterCount) {
                    gcapp.showImage();
                    cout << iterCount << ">" << endl;
                } else {
                    cout << "rect must be determined>" << endl;
                }
                break;
            }
            default:
                break;
        }
    }

    exit_main:
    destroyWindow(winName);
    return 0;
}
