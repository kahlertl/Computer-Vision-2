#ifndef CV2_PATCHMATCH_HPP
#define CV2_PATCHMATCH_HPP

#include <opencv2/opencv.hpp>

class PatchMatch
{
    int nrows;
    int ncols;
    int niterations;

    // parameters
    const int maxoffset;
    const int match_radius;
    const int iterations;
    const float search_ratio;
    const bool max_search_radius;
    int search_radius;

    int border;

    cv::Mat result;
    cv::Mat quality;

    void initialize(const cv::Mat& image1, const cv::Mat& image2);

    void propagate(const cv::Mat& image1, const cv::Mat& image2);

    void random_search(const cv::Mat& image1, const cv::Mat& image2);

public:

    PatchMatch(int maxoffset, int match_radius, int iterations = 3, float search_ratio = 0.5, int search_radius = -1);

    void match(const cv::Mat& image1, const cv::Mat& image2, cv::Mat& result);
};

void flow2rgb(const cv::Mat& flow, cv::Mat& rgb);

#endif //CV2_PATCHMATCH_HPP
