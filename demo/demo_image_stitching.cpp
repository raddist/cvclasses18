/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

static void put_points_text(cv::Mat image, size_t points)
{
    const auto fontScale = 0.5;
    const auto thickness = 1;
    const auto color = cv::Scalar(0, 255, 0);
    std::stringstream ss;
    ss << "points: " << points;
    cv::putText(image, ss.str(), cv::Point{2, 15}, CV_FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, 8, false);
}

static void threshold_change(int value, void* ptr)
{
    auto& detector = *(cv::Ptr<cvlib::corner_detector_fast>*)(ptr);
    detector->setThreshold(value);
}

static void ratio_change(int value, void* ptr)
{
    auto& matcher = *(cv::Ptr<cvlib::descriptor_matcher>*)(ptr);
    matcher->setRatio(value / 10.0f);
}

int demo_image_stitching(int argc, char* argv[])
{
	cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_matching_wnd = "demo matching";
    const auto demo_stitching_wnd = "demo stitching";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_matching_wnd);
    cv::namedWindow(demo_stitching_wnd);

    // auto detector = cvlib::corner_detector_fast::create();
    auto detector = cv::BRISK::create();
    //auto matcher = cvlib::descriptor_matcher();
    auto matcher = cv::BFMatcher();
    //auto stitcher = cvlib::stitcher();
    //auto stitcher = cv::Stitcher();

    int ratio = 15;
    int max_distance = 100;
    int threshold = 40;
    cv::createTrackbar("max distance", main_wnd, &max_distance, 500);
    cv::createTrackbar("ratio", main_wnd, &ratio, 30, ratio_change, &matcher);
    cv::createTrackbar("threshold", main_wnd, &threshold, 255, threshold_change, &detector);

    /// \brief helper struct for tidy code
    struct img_features
    {
        cv::Mat img;
        std::vector<cv::KeyPoint> corners;
        cv::Mat descriptors;
    };

    img_features ref;
    img_features test;
    std::vector<std::vector<cv::DMatch>> pairs;
    std::vector<std::vector<cv::DMatch>> good_pairs;

    cv::Mat frame_gray;
    cv::Mat main_frame;
    cv::Mat demo_matching_frame, demo_stitching_frame;
    cv::Mat homography;

    utils::fps_counter fps;
    int pressed_key = 0;
    bool init = false;
    while (pressed_key != 27) // ESC
    {
        cap >> test.img;

        cv::cvtColor(test.img, frame_gray, CV_BGR2GRAY);
        detector->detect(frame_gray, test.corners);
        detector->compute(frame_gray, test.corners, test.descriptors);
        cv::drawKeypoints(test.img, test.corners, main_frame);
        cv::imshow(main_wnd, main_frame);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ') // space
        {
            //cv::resize(test.img, test.img, cv::Size(320, 240));

            if (!init)
            {
                cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, test.img.cols * 1.5, 0, 1, test.img.rows * 0.5);
                cv::warpAffine(test.img, test.img, trans_mat, cv::Size(4 * test.img.cols, 2 * test.img.rows));
                test.img.copyTo(ref.img);
                init = true;
            }
            else
            {
                cv::cvtColor(ref.img, frame_gray, CV_BGR2GRAY);
                detector->detect(frame_gray, ref.corners);
                detector->compute(frame_gray, ref.corners, ref.descriptors);

                matcher.knnMatch(test.descriptors, ref.descriptors, pairs, 2);
                //matcher.radiusMatch(test.descriptors, ref.descriptors, pairs, max_distance);

                std::vector<cv::Point2f> pts_from, pts_to;

                for (const auto& match : pairs)
                {
                    if (match[0].distance / match[1].distance < 0.5)
                    {
                        good_pairs.emplace_back(match);
                        pts_from.emplace_back(test.corners[match[0].queryIdx].pt);
                        pts_to.emplace_back(ref.corners[match[0].trainIdx].pt);
                    }
                }

                cv::drawMatches(test.img, test.corners, ref.img, ref.corners, good_pairs, demo_matching_frame);
                cv::imshow(demo_matching_wnd, demo_matching_frame);

                if (pts_from.empty() || pts_to.empty())
                {
                    continue;
                }

                homography = cv::findHomography(pts_from, pts_to, cv::RANSAC);

                cv::Mat warp_image;

                if (homography.empty())
                {
                    continue;
                }

                cv::warpPerspective(test.img, warp_image, homography, ref.img.size());

                for (int j = 0; j < ref.img.cols; j++)
                {
                    for (int i = 0; i < ref.img.rows; i++)
                    {
                        cv::Vec3b color_ref = ref.img.at<cv::Vec3b>(cv::Point(j, i));
                        cv::Vec3b color_warp = warp_image.at<cv::Vec3b>(cv::Point(j, i));

                        if (cv::norm(color_ref) == 0)
                            ref.img.at<cv::Vec3b>(cv::Point(j, i)) = color_warp;
                    }
                }
            }

            pairs.clear();
            good_pairs.clear();

            utils::put_fps_text(ref.img, fps);
            cv::imshow(demo_stitching_wnd, ref.img);
        }
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_matching_wnd);
    cv::destroyWindow(demo_stitching_wnd);

    return 0;
}
