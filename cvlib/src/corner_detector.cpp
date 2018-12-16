/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace
{
const int kernel_size = 7;
const int desc_pairs = 8 * 32;

void initialize_matched_pairs(std::vector<std::pair<cv::Point, cv::Point>>& pairs)
{
    int min_coord = -3;
    cv::RNG rnd;
    for (int i = 0; i < desc_pairs; ++i)
    {
        cv::Point p1(rnd.uniform(min_coord, min_coord + kernel_size),
                     rnd.uniform(min_coord, min_coord + kernel_size));
        cv::Point p2(rnd.uniform(min_coord, min_coord + kernel_size),
                     rnd.uniform(min_coord, min_coord + kernel_size));
        pairs.emplace_back(p1, p2);
    }
}
}

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
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

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{

    if (matched_pairs_.size() == 0)
    {
        initialize_matched_pairs(matched_pairs_);
    }

    const int desc_length = 32;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_8U);

    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    cv::Mat image_mat = image.getMat();
    if (image_mat.channels() == 3)
        cv::cvtColor(image_mat, image_mat, cv::COLOR_BGR2GRAY);

    uint8_t* ptr = reinterpret_cast<uint8_t*>(desc_mat.ptr());

    for (const auto& pt : keypoints)
    {
        int descriptor = 0;
        int x = (int)pt.pt.x;
        int y = (int)pt.pt.y;
        cv::Point center{x, y};

        int counter = 0;
        for (auto && p_pair : matched_pairs_)
        {
            counter++;

            uchar p1_brightness = image_mat.at<uchar>(center + p_pair.first);
            uchar p2_brightness = image_mat.at<uchar>(center + p_pair.second);

            descriptor = descriptor << 1;
            if (p1_brightness <= p2_brightness)
            {
                descriptor |= 1;
            }

            if (counter % (sizeof(uint8_t) * 8) == 0)
            {
                *ptr = descriptor;
                ptr++;
            }
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray frame, cv::InputArray,
    std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool /*= false*/)
{
    set_threshold(30);
    detect(frame, keypoints);
    compute(frame, keypoints, descriptors);

    std::cout << " \n\n";
}

} // namespace cvlib
