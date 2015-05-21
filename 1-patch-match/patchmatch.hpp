#ifndef CV2_PATCHMATCH_HPP
#define CV2_PATCHMATCH_HPP

#include <opencv2/opencv.hpp>
#include <limits>

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

    cv::Mat flow;

    void initialize(const cv::Mat& image1, const cv::Mat& image2);

    void propagate(const cv::Mat& image1, const cv::Mat& image2, const int row, const int col);

    void random_search(const cv::Mat& image1, const cv::Mat& image2, const int row, const int col);

    /**
     * Creates a random point in the interval [-1, 1] x [-1, 1]
     */
    inline cv::Point2f random_interval()
    {
        return cv::Point2f(0, 0);

        // return cv::Point2f(
        //     (float) std::rand() / ((float) std::RAND_MAX / 2) - 1.0, // x
        //     (float) std::rand() / ((float) std::RAND_MAX / 2) - 1.0  // y
        // );
    }

public:

    PatchMatch(int maxoffset, int match_radius, int iterations = 3, float search_ratio = 0.5, int search_radius = -1);

    void match(const cv::Mat& image1, const cv::Mat& image2, cv::Mat& result);
};

void flow2rgb(const cv::Mat& flow, cv::Mat& rgb);
float ssd(const cv::Mat& image1, const cv::Point2i& center1, const cv::Mat& image2, const cv::Point2i& center2, const int radius, const float halt = std::numeric_limits<float>::infinity());

#endif //CV2_PATCHMATCH_HPP
