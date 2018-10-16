 /* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <vector>

namespace
{
using subimage = std::tuple<cv::Mat, cv::Range, cv::Range>;
using subimages = std::vector<subimage>;

class MergingImg
{
public:
    MergingImg(subimages& splittedImgs, int ind)
        : m_splittedImgs{ splittedImgs }
        , m_inds{ind}
        , m_isExpired{ false }
        , m_mean{-1}
    {
    }

    MergingImg(const MergingImg& other) = default;
    MergingImg& operator=(const MergingImg& other) = default;
    MergingImg& operator=(const MergingImg&& other)
    {
        m_splittedImgs = other.m_splittedImgs;
        m_inds = other.m_inds;
        m_isExpired = other.m_isExpired;
        m_mean = other.m_mean;
        return *this;
    }

    bool GetIsExpired() const { return m_isExpired; };
    void SetIsExpired(bool expired) { m_isExpired = expired; };

    int GetMean() const
    {
        if (m_mean != -1)
        {
            return m_mean;
        }

        int result = 0;
        int divisor = 0;
        for ( auto && ind : m_inds )
        {
            int size = std::get<0>(m_splittedImgs[ind]).cols
                     * std::get<0>(m_splittedImgs[ind]).rows;
            result += (cv::mean(std::get<0>(m_splittedImgs[ind]))[0] * size);
            divisor += size;
        }
        return static_cast<int>(std::floor(result / divisor));
    }

    void AddImg(MergingImg& img)
    {
        std::vector<int> tmp = img.GetInds();
        m_inds.insert(m_inds.end(), tmp.begin(), tmp.end() );
        m_mean = -1;
    }

    void Merge()
    {
        for (auto && ind : m_inds)
        {
            std::get<0>(m_splittedImgs[ind]).setTo( GetMean() );
        }
    }

    bool IsNeighbour(MergingImg& img)
    {
        std::vector<int> tmp = img.GetInds();
        for (auto && l_ind : m_inds)
        {
            cv::Range l_yRng = std::get<1>(m_splittedImgs[l_ind]);
            cv::Range l_xRng = std::get<2>(m_splittedImgs[l_ind]);
            for (auto && r_ind : tmp)
            {
                cv::Range r_yRng = std::get<1>(m_splittedImgs[r_ind]);
                cv::Range r_xRng = std::get<2>(m_splittedImgs[r_ind]);

                if ( (l_xRng.start <= r_xRng.end && r_xRng.start <= l_xRng.end)
                    && (l_yRng.start <= r_yRng.end && r_yRng.start <= l_yRng.end) )
                {
                    return true;
                }
            }
        }
        return false;
    }

    std::vector<int>& GetInds() { return m_inds; };

private:
    std::vector<int> m_inds;
    subimages& m_splittedImgs;

    bool m_isExpired;
    int m_mean = -1;
};

void split_image(cv::Mat image, cv::Range yr, cv::Range xr,  double stddev, subimages& results)
{
    cv::Mat subImg{ image(cv::Range(yr.start, yr.end), cv::Range(xr.start, xr.end)) };
    cv::Mat mean;
    cv::Mat dev;
    cv::meanStdDev(subImg, mean, dev);

    if (dev.at<double>(0) <= stddev)
    {
        results.emplace_back(subImg, yr, xr );
        subImg.setTo(mean);
        return;
    }

    const auto width = (xr.end - xr.start) / 2;
    const auto height = (yr.end - yr.start) / 2;

    split_image(image, cv::Range(yr.start, yr.start + height), cv::Range(xr.start, xr.start + width), stddev, results);
    split_image(image, cv::Range(yr.start, yr.start + height), cv::Range(xr.start + width, xr.end), stddev, results);
    split_image(image, cv::Range(yr.start + height, yr.end), cv::Range(xr.start + width, xr.end), stddev, results);
    split_image(image, cv::Range(yr.start + height, yr.end), cv::Range(xr.start, xr.start + width), stddev, results);
}
}

void merge_image(cv::Mat image, double stddev, std::vector<MergingImg>& mergeIMGs)
{
    bool mergeMore = false;
    for (int i = 0; i < mergeIMGs.size(); ++i)
    {
        if (mergeIMGs[i].GetIsExpired())
            continue;

        int imgMean = mergeIMGs[i].GetMean();
        for (int j = i + 1; j < mergeIMGs.size(); ++j)
        {
            if (mergeIMGs[j].GetIsExpired())
                continue;

            if ((std::abs)(imgMean - mergeIMGs[j].GetMean()) <= stddev
                && mergeIMGs[i].IsNeighbour(mergeIMGs[j]) )
            {
                mergeIMGs[i].AddImg(mergeIMGs[j]);
                mergeIMGs[j].SetIsExpired(true);

                mergeMore = true;
            }
        }

        if (mergeMore)
        {
            break;
        }
    }


    //if (mergeMore)
    //{
    //    auto new_end = std::remove_if(mergeIMGs.begin(), mergeIMGs.end(), [](const MergingImg& img) {return img.GetIsExpired(); });
    //    mergeIMGs.erase(new_end, mergeIMGs.end());

    //    merge_image(image, stddev, mergeIMGs);
    //}
    //else
    {
        for (auto & img : mergeIMGs)
        {
            img.Merge();
        }
    }
}

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    cv::Mat res = image;

    // split part
    subimages splittedImgs{};
    split_image(res, cv::Range(0, res.rows), cv::Range(0, res.cols), stddev, splittedImgs);

    // merge part
    std::vector<MergingImg> mergeIMGs;
    for ( int i = 0; i < splittedImgs.size(); ++i)
    {
        if (std::get<0>(splittedImgs[i]).cols != 0
            && std::get<0>(splittedImgs[i]).rows != 0)
        {
            mergeIMGs.emplace_back(splittedImgs, i);
        }
    }
    merge_image(res, stddev, mergeIMGs);
    return res;
}
} // namespace cvlib
