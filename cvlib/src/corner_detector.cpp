/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

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
    const int desc_length = 3*3 + 4;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);

    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    cv::Mat image_mat = image.getMat();
    if (image_mat.channels() == 3)
        cv::cvtColor(image_mat, image_mat, cv::COLOR_BGR2GRAY);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());

    for (const auto& pt : keypoints)
    {
        int x = (int)pt.pt.x;
        int y = (int)pt.pt.y;
        cv::Point center{x, y};

        double min = 1.0;
        double max = 255.0;
        std::vector<int> radiuses{1, 2, 3};
        for (auto && radius : radiuses)
        {
            cv::Mat region = image_mat(cv::Range(y - radius, y + radius + 1), cv::Range(x - radius, x + radius + 1));

            cv::minMaxLoc(region, &min, &max);

            cv::Mat mmean, mstd;
            cv::meanStdDev(region, mmean, mstd);
            float mean = (float)mmean.at<double>(0);
            float std = (float)mstd.at<double>(0);

            *ptr = int(mean / max * 1e4);
            ptr++;
            *ptr = int(std / max * 1e4);
            ptr++;

            cv::Moments mom = cv::moments(region);
            double hu[7];
            cv::HuMoments(mom, hu);
            *ptr = int(hu[0] * 10000);
            ptr++;
        }

        // calculate direction with max brightness difference
        int center_brightness = image_mat.at<unsigned char>(center);
        int max_direction = 0;
        int max_diff = 0;
        for (int j = 0; j < 16; ++j)
        {
            uchar circle_point_brightness = image_mat.at<uchar>(center + circle_template_[j]);
            int diff = std::abs(circle_point_brightness - center_brightness);
            if (diff > max_diff)
            {
                max_diff = diff;
                max_direction = j;
            }
        }

        // store diff with forward direction
        *ptr = int(max_diff / max * 1e4);
        ptr++;

        // store diff with wright direction
        int wright_direction = (max_direction + 4) % 16;
        uchar circle_write_brightness = image_mat.at<uchar>(center + circle_template_[wright_direction]);
        *ptr = int(circle_write_brightness / max * 1e4);
        ptr++;

        // store diff with backward direction
        int back_direction = (max_direction + 4) % 16;
        uchar circle_back_brightness = image_mat.at<uchar>(center + circle_template_[back_direction]);
        *ptr = int(circle_back_brightness / max * 1e4);
        ptr++;

        // store diff with left direction
        int left_direction = (max_direction + 4) % 16;
        uchar circle_left_brightness = image_mat.at<uchar>(center + circle_template_[left_direction]);
        *ptr = int(circle_left_brightness / max * 1e4);
        ptr++;
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray frame, cv::InputArray,
    std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool /*= false*/)
{
    std::cout << keypoints.size() << " \n";
    set_threshold(20);
    detect(frame, keypoints);
    std::cout << keypoints.size() << " \n";
    compute(frame, keypoints, descriptors);
    std::cout << keypoints.size() <<  " \n";
    std::cout << " \n\n";
}

} // namespace cvlib
