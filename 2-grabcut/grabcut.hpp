#ifndef CV2_GRABCUT_HPP
#define CV2_GRABCUT_HPP

#include <opencv2/imgproc/imgproc.hpp>

namespace cv
{
/**
 * Modified version of the GrabCut algorithm that takes only
 * those pixels for foreground distribution that are most likely
 * in the background distribution.
 *
 * The portion of that pixels can be modified by the "tolerance"
 * parameter.
 */
void grabCut(InputArray _img, InputOutputArray _mask, Rect rect,
             InputOutputArray _bgdModel, InputOutputArray _fgdModel,
             int iterCount, double tolerance, int mode = GC_EVAL);

}


#endif //CV2_GRABCUT_HPP
