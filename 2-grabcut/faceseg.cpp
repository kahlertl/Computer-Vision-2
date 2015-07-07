#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "argtable2.h"

#include "grabcut.hpp"

using namespace std;
using namespace cv;


// Defaults for the Extended GraphCut parameters
int iterations      = 3;     // number iterations
double tolerance    = 0.5;
int neighbors       = GC_N8;
bool extended       = false;
double connectivity = 25;
double contrast     = 10;

const Scalar BLUE = Scalar(255, 0, 0);
const Scalar RED  = Scalar(0, 0, 255);

// drawing parameters
const int thickness  = 2;   // Thickness of the ellipse arc outline, if positive.
                            // Otherwise, this indicates that a filled ellipse sector is to be drawn.
const int shift      = 0;   // Number of fractional bits in the coordinates of the center and values of axes
const int angle      = 0;   // Ellipse rotation angle in degrees.
const int startAngle = 0;   // Starting angle of the elliptic arc in degrees.
const int endAngle   = 360; // Ending angle of the elliptic arc in degrees.
const int connected  = 8;

static void wait(const string& message = "Press ESC to continue ...")
{
    cerr << message << endl;
    while (true) {
        if ((uchar) waitKey(0) == '\x1b') {
            break;
        }
    }
}

void displaySegmentation(const Mat& image, const Mat& mask)
{
    Mat binMask;
    binMask.create(mask.size(), CV_8UC1);
    binMask = mask & 1;

    Mat segmentation;
    image.copyTo(segmentation, binMask);

    imshow("Segmentation", segmentation);
}

static void sanitizeRectangular(const Mat& image, Rect& rect)
{
    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, image.cols - rect.x);
    rect.height = min(rect.height, image.rows - rect.y);
}

void segmentFace(const Mat& image, Mat& mask, Mat& canvas, Rect& face, vector<Rect>& eyes)
{

    Mat backgroundModel, foregroundModel;

    // scale face region with 1.5
    Rect rect = face;
    rect.x -= rect.width  / 4;
    rect.y -= rect.height / 4;
    rect.width  *= 1.5;
    rect.height *= 1.5;

    sanitizeRectangular(image, rect);

    // paint rectangular around the face region
    rectangle(canvas, rect, BLUE, thickness, connected, shift);

    // initialize GrabCut mask
    // the face region is the typical GrabCut rectangular
    mask.create(image.size(), CV_8UC1);
    mask.setTo(GC_BGD);
    (mask(rect)).setTo(Scalar(GC_PR_FGD));

    // Each tracked eye will set hard to foreground
    for (int i = 0; i < eyes.size(); i++) {
        rect = eyes[i];

        // move rectangular to correct global position
        // in the image
        rect.x += face.x;
        rect.y += face.y;

        // decrease size of the eye rectangular
        rect.x += rect.width / 4;
        rect.y += rect.height / 4;
        rect.width  *= 0.75;
        rect.height *= 0.75;

        // inpaint eye regions as circles
        rectangle(canvas, rect, RED, thickness, connected, shift);

        (mask(rect)).setTo(Scalar(GC_FGD));
    }

    cerr << "Perform GrabCut ... ";
    extendedGrabCut(image, mask, face, backgroundModel, foregroundModel, iterations,
                    tolerance, extended, connectivity, contrast, neighbors, GC_INIT_WITH_MASK);
    cerr << "Done" << endl;

}

int run(const Mat& image, CascadeClassifier& faceCascade, CascadeClassifier& eyesCascade)
{
    Mat grayImage, canvas, finalMask;

    // initialize canvas with original image
    image.copyTo(canvas);

    // the final mask is the union of all found segmentations
    finalMask.create(image.size(), CV_8UC1);
    finalMask.setTo(GC_BGD);

    cvtColor(image, grayImage, CV_BGR2GRAY);
    equalizeHist(grayImage, grayImage);

    cerr << "Detect faces ... ";
    vector<Rect> faces;
    faceCascade.detectMultiScale(grayImage, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    cerr << "Done" << endl;

    // inpaint the found regions
    for (int i = 0; i < faces.size(); i++) {
        cerr << "Face " << i << " ... " << endl;
        Mat mask;
        // get region of interest (RIO) for eye detection
        Mat faceROI = grayImage(faces[i]);
        std::vector<Rect> eyes;

        // In each face, detect eyes
        eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

        segmentFace(image, mask, canvas, faces[i], eyes);
        finalMask |= mask;

        // give OS time to display images
        waitKey(300);
    }

    namedWindow("Segmentation", WINDOW_AUTOSIZE);
    namedWindow("Detection", WINDOW_AUTOSIZE);

    imshow("Detection", canvas);
    displaySegmentation(image, finalMask);

    wait("Press ESC to exit ...");

    return 0;
}

int main(int argc, char *argv[])
{
    CascadeClassifier faceClassifier, eyesClassifier;
    Mat image;

    // Command line options and arguments
    struct arg_lit*  help         = arg_lit0("h", "help",                      "Show this help message");
    struct arg_lit*  version      = arg_lit0("v", "version",                   "Print version information and exit");
    struct arg_file* faceCascade  = arg_file0("f", "face-cascade", nullptr,    "XML-file of the face cascade");
    struct arg_file* eyesCascade  = arg_file0("e", "eye-cascade", nullptr,     "XML-file of the eye cascade");
    struct arg_file* infile       = arg_filen(nullptr, nullptr, "image", 1, 1, "input image");
    struct arg_end  *end          = arg_end(20);

    void* argtable[] = { help, version, faceCascade, eyesCascade, infile, end };

    const char* progname = "faceseg";

    int nerrors;
    int exitcode = 0;

    // verify the argtable[] entries were allocated sucessfully
    if (arg_nullcheck(argtable) != 0) {
        // null pointer entries were detected, some allocations must have failed
        printf("%s: insufficient memory\n", progname);
        exitcode = 1;
        goto exit;
    }

    // set any command line default values prior to parsing
    faceCascade->filename[0] = "haarcascade_frontalface_alt.xml";
    eyesCascade->filename[0]  = "haarcascade_eye_tree_eyeglasses.xml";

    // Parse the command line as defined by argtable[]
    nerrors = arg_parse(argc, argv, argtable);

    // special case: '--help' takes precedence over error reporting
    if (help->count > 0) {
        cout << "Usage: " << progname << " [options] image" << endl
             << endl;

        #ifdef __linux__
        cout << "Default default location for openCV cascades:" << endl
            << endl
            << "    /usr/share/opencv/haarcascades/" << endl
            << endl;
         #endif

        arg_print_glossary(stdout, argtable, "  %-25s %s\n");

        exitcode = 0;

        goto exit;
    }

    // special case: '--version' takes precedence error reporting
    if (version->count > 0) {
        // printf("'%s' example program for the \"argtable\" command line argument parser.\n", progname);
        cout << __DATE__ << ", Lucas Kahlert" << endl;
        exitcode = 0;
        goto exit;
    }

    // If the parser returned any errors then display them and exit
    if (nerrors > 0) {
        // Display the error details contained in the arg_end struct.
        arg_print_errors(stdout, end, progname);

        printf("Try \"%s --help\" for more information.\n",progname);

        exitcode = 1;
        goto exit;
    }

    // special case: uname with no command line options induces brief help */
    if (argc == 1) {
        printf("Try \"%s --help\" for more information.\n", progname);
        exitcode = 0;
        goto exit;
    }

    // try to load cascade files
    if( !faceClassifier.load(faceCascade->filename[0])){
        cerr << "Error: can not load face cascade \"" << faceCascade->filename[0] << "\"" << endl;
        return 1;
    };
    if( !eyesClassifier.load(eyesCascade->filename[0])){
        cerr << "Error: can not load eye cascade \"" << eyesCascade->filename[0] << "\"" << endl;
        return 1;
    };

    // try to read input image
    image = imread(*(infile->filename), CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
        cerr << "Error: Cannot read '" << *(infile->filename) << "'" << endl;

        exitcode = 1;
        goto exit;
    }

    run(image, faceClassifier, eyesClassifier);

    exit:
    // deallocate each non-null entry in argtable[]
    arg_freetable(argtable,sizeof(argtable)/sizeof(argtable[0]));

    return 0;
}
