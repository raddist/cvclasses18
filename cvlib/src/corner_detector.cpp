/* FAST corner detector algorithm implementation.
* @file
* @date 2018-10-16
* @author Anonymous
*/

#include "cvlib.hpp"

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
   return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints,
   cv::InputArray /*mask = cv::noArray()*/)
{
   keypoints.clear();
   
   if (image.empty())
      return;

   else if (image.channels() == 1)
      image.copyTo(image_);

   else if (image.channels() == 3)
      cv::cvtColor(image, image_, cv::COLOR_BGR2GRAY);

   for (auto y = 3; y < image_.rows - 3; ++y)
      for (auto x = 3; x < image_.cols - 3; ++x)
         if (is_keypoint(cv::Point(x, y), 4, 3) &&
            is_keypoint(cv::Point(x, y), 1, 12))
            keypoints.push_back(cv::KeyPoint(cv::Point(x, y), 1.f));
}

void corner_detector_fast::set_threshold(int thresh)
{
   threshold_ = thresh;
}

brightness_check_result corner_detector_fast::check_brightness(unsigned int circle_point_num, cv::Point center)
{
   uchar center_brightness = image_.at<uchar>(center);
   uchar circle_point_brightness = image_.at<uchar>(center + circle_template_[circle_point_num]);
   if (circle_point_brightness <= center_brightness - static_cast<uchar>(threshold_))
      return brightness_check_result::darker;
   else if (center_brightness + static_cast<uchar>(threshold_) <= circle_point_brightness)
      return brightness_check_result::brighter;
   else
      return brightness_check_result::similar;
}

bool corner_detector_fast::is_keypoint(cv::Point center, unsigned int step, unsigned int num)
{
   std::vector<unsigned int> check_results(3, 0);

   unsigned int prev_check_res = check_brightness(0, center);
   check_results[prev_check_res]++;
   for (int i = 1; i < 16*2; i += step)
   {
       unsigned int cur_check_res = check_brightness(i%16, center);
       check_results[cur_check_res]++;
       if (cur_check_res == prev_check_res)
       {
           if (check_results[cur_check_res] >= num)
               break;
       }
       else
       {
           check_results[prev_check_res] = 0;
           if (i >= 16)
               break;
       }
   }

   return ((check_results[brightness_check_result::brighter] >= num)
        || (check_results[brightness_check_result::darker]   >= num));
}
} // namespace cvlib
