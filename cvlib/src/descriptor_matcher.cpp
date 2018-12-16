/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <chrono>
#include <thread>

namespace cvlib
{
void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k /*unhandled*/,
                                      cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];

    matches.resize(q_desc.rows);

    cv::RNG rnd;
    for (int i = 0; i < q_desc.rows; ++i)
    {
        // \todo implement Ratio of SSD check.
        //matches[i].emplace_back(i, rnd.uniform(0, t_desc.rows), FLT_MAX);
    }
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    // \todo implement matching with "maxDistance"
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];
    matches.resize(q_desc.rows);

    std::vector<int> trueMatches; // good matches for knnMatch
    std::vector<double> distances;

    for (auto i = 0; i < q_desc.rows; ++i)
    {
        for (auto j = 0; j < t_desc.rows; ++j)
        {
            cv::Mat diff;
            cv::absdiff(q_desc.row(i), t_desc.row(j), diff);
            double ssd_dist = cv::norm(diff, cv::NORM_L2);

            //std::cout << q_desc.row(i) << std::endl;
            //std::cout << t_desc.row(j) << std::endl;
            //std::cout << diff << std::endl;
            //std::cout << ssd_dist << std::endl;
            //std::this_thread::sleep_for(std::chrono::milliseconds(10000));


            if (ssd_dist < maxDistance)
            {
                trueMatches.push_back(j);
                distances.push_back(ssd_dist);
            }
        }

        int nearestPoint = 0;
        if (!trueMatches.empty())
        {
            int minDistIndex = std::min_element(distances.begin(), distances.end()) - distances.begin();
            double minDist = *std::min_element(distances.begin(), distances.end());

            nearestPoint = trueMatches[minDistIndex];

            bool truePair = true;
            for (auto k = 0; k < trueMatches.size(); k++)
            {
                if (trueMatches.at(k) != nearestPoint && minDist / distances.at(k) > 0.001)
                {
                    truePair = false;
                    std::cout << ratio_ << std::endl;
                    break;
                }
            }

            if (truePair)
            {
                matches[i].emplace_back(i, nearestPoint, (float)minDist);
                /*std::cout << q_desc.row(i) << std::endl;
                std::cout << t_desc.row(nearestPoint) << std::endl;
                std::cout << minDist << std::endl;*/
                //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }
        trueMatches.clear();
        distances.clear();
    }

    //knnMatchImpl(queryDescriptors, matches, 1, masks, compactResult);
}
} // namespace cvlib
