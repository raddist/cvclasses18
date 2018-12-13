/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <iostream>

namespace cvlib
{
void motion_segmentation::setVarThreshold(double threshold)
{
    var_threshold_ = threshold;
}

void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double leaning_rate)
{
    // \todo implement your own algorithm:
    //       * MinMax
    //       * Mean
    //       * 1G
    //       * GMM

    // \todo implement bg model updates
    if (_image.empty())
    {
        _image.copyTo(_fgmask);
        return;
    }

    cv::Mat converted;
    cv::Mat image;
    _image.getMat().convertTo(converted, CV_32F);
    cv::medianBlur(converted, image, 5);

    if (current_frame_ == 0)
    {
        bg_model_ = cv::Mat::zeros(image.size(), CV_32F);
    }

    ++current_frame_;

    cv::Mat diff;
    cv::absdiff(image, bg_model_, diff);
    cv::accumulateWeighted(image, bg_model_, leaning_rate);

    cv::Mat result = (diff > var_threshold_) * 255;
    _fgmask.assign(result);
}
} // namespace cvlib
