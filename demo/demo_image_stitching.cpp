/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_image_stitching(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    //auto detector = cvlib::corner_detector_fast::create();
    auto detector = cv::ORB::create();
    auto matcher = cv::BFMatcher();
    auto stitcher = cvlib::Stitcher();

    int detector_threshold = 30;
    int ratio = 12;
    int max_distance = 20;

    //detector->setThreshold(detector_threshold);
    //cv::createTrackbar(
    //    "Threshold", main_wnd, &detector_threshold, 255,
    //    [](int threshold, void* detector) { ((cvlib::corner_detector_fast*)detector)->setThreshold(threshold); }, (void*)detector);

    //matcher.set_ratio(ratio / 10.0f);
    //cv::createTrackbar(
    //    "Ratio * 10", main_wnd, &ratio, 20, [](int ratio, void* matcher) { ((cvlib::descriptor_matcher*)matcher)->set_ratio(ratio / 10.0f); },
    //    (void*)&matcher);

    cv::createTrackbar("MaxDist", main_wnd, &max_distance, 500);

    cv::Mat refImg, testImg;
    std::vector<cv::KeyPoint> refCorners, testCorners;
    cv::Mat refDescriptors, testDescriptors;
    std::vector<std::vector<cv::DMatch>> pairs;

    cv::Mat main_frame;
    cv::Mat demo_frame;
    utils::fps_counter fps;
    int pressed_key = 0;
    bool needToInit = true;

    while (pressed_key != 27) // ESC
    {
        cap >> testImg;
        testImg.copyTo(main_frame);
        utils::put_fps_text(main_frame, fps);
        cv::imshow(main_wnd, main_frame);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ') // space
        {
            if (needToInit)
            {
                testImg.copyTo(refImg);
                detector->detectAndCompute(refImg, cv::Mat(), refCorners, refDescriptors);
                needToInit = false;
            }
            else
            {
                stitcher.stitch(testCorners, refCorners, testDescriptors, refDescriptors, pairs, refImg);
            }
        }

        if (refCorners.empty())
            continue;

        detector->detectAndCompute(testImg, cv::Mat(), testCorners, testDescriptors);

        matcher.radiusMatch(testDescriptors, refDescriptors, pairs, max_distance);

        stitcher.makeStitchedImg(testImg, testCorners, refImg, refCorners, pairs, demo_frame);

        cv::imshow(demo_wnd, demo_frame);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
