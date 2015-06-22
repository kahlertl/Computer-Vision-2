#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <getopt.h> // getopt_long()

using namespace std;
using namespace cv;

const string windowName  = "Face detection";

const Scalar BLUE = Scalar(255, 0, 0);
const Scalar RED  = Scalar(0, 0, 255);

// command line defaults
string faceCascadeName = "haarcascade_frontalface_alt.xml";
string eyesCascadeName = "haarcascade_eye_tree_eyeglasses.xml";

// command line option list
static const struct option long_options[] = {
    {"help",         no_argument,       0, 'h'},
    {"face-cascade", required_argument, 0, 'f'},
    {"eye-cascade" , required_argument, 0, 'e'},
    0 // end of parameter list
};

static void usage()
{
    cout << "Usage: facedetector [options] image" << endl
    << endl
    << "This program will find and segment a faces from a given image" << endl
    << endl
    #ifdef __linux__
    << "Default default location for openCV cascades:" << endl
    << endl
    << "    /usr/share/opencv/haarcascades/" << endl
    << endl
    #endif
    << "  options:" << endl
    << "    -h, --help            Show this help message" << endl
    << "    -f, --face-cascade    XML-file of the face cascade" << endl
    << "                          Default: " << faceCascadeName << endl
    << "    -e, --eye-cascade     XML-file of the eye cascade" << endl
    << "                          Default: " << eyesCascadeName << endl
    << endl;
}

static bool parsePositionalImage(Mat &image, const int channels, const string &name, int argc, char const *argv[])
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

int main(int argc, char const *argv[])
{
    CascadeClassifier faceCascade;
    CascadeClassifier eyesCascade;
    Mat image;
    Mat grayImage;
    Mat canvas;

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

            case '?': // missing option
                return 1;

            default: // unknown
                cerr << "unknown parameter: " << optarg << endl;
                break;
        }
    }

    // load remaining command line argument
    if (!parsePositionalImage(image, CV_LOAD_IMAGE_COLOR, "image", argc, argv)) {
        return 1;
    }
    if( !faceCascade.load(faceCascadeName)){
        cerr << "Error: can not load face cascade \"" << faceCascadeName << "\"" << endl;
        return 1;
    };
    if( !eyesCascade.load(eyesCascadeName)){
        cerr << "Error: can not load eye cascade \"" << eyesCascadeName << "\"" << endl;
        return 1;
    };

    std::vector<Rect> faces;
    Mat frame_gray;

    image.copyTo(canvas);
    cvtColor(image, grayImage, CV_BGR2GRAY);
    // equalizeHist(grayImage, grayImage);

    faceCascade.detectMultiScale(grayImage, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    // drawing parameters
    const int thickness  = 4;   // Thickness of the ellipse arc outline, if positive.
                                // Otherwise, this indicates that a filled ellipse sector is to be drawn.
    const int shift      = 0;   // Number of fractional bits in the coordinates of the center and values of axes
    const int angle      = 0;   // Ellipse rotation angle in degrees.
    const int startAngle = 0;   // Starting angle of the elliptic arc in degrees.
    const int endAngle   = 360; // Ending angle of the elliptic arc in degrees.
    const int connected  = 8;

    // inpaint the found regions
    for (int i = 0; i < faces.size(); i++) {
        // draw ellipse around face region
        Point center(faces[i].x + faces[i].width  / 2,
                     faces[i].y + faces[i].height / 2);
        Size axes(faces[i].width / 2,
                  faces[i].height / 2);
        ellipse(canvas, center, axes, angle, startAngle, endAngle, BLUE, thickness, connected, shift);

        // get region of interest (RIO) for eye detection
        Mat faceROI = grayImage(faces[i]);
        std::vector<Rect> eyes;

        // In each face, detect eyes
        eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

        // inpaint eye regions as circles
        for (int j = 0; j < eyes.size(); j++) {
            center.x = faces[i].x + eyes[j].x + eyes[j].width / 2;
            center.y = faces[i].y + eyes[j].y + eyes[j].height / 2;

            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(canvas, center, radius, RED, thickness, connected, shift);
        }
    }
    imshow(windowName, canvas);

    cerr << "Press ESC to exit ..." << endl;
    while (true) {
        if ((uchar) waitKey(0) == '\x1b') {
            break;
        }
    }

    return 0;
}
