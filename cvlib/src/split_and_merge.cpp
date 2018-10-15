 /* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <vector>

namespace
{
using images = std::vector<cv::Mat>;

class MergingImg
{
public:
    MergingImg(images& splittedImgs, int ind)
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
            int size = m_splittedImgs[ind].size().height
                     * m_splittedImgs[ind].size().width;
            result += (cv::mean(m_splittedImgs[ind])[0] * size);
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
            m_splittedImgs[ind].setTo( GetMean() );
        }
    }

    std::vector<int>& GetInds() { return m_inds; };

private:
    std::vector<int> m_inds;
    images& m_splittedImgs;

    bool m_isExpired;
    int m_mean = -1;
};

void split_image(cv::Mat image, double stddev, images& results)
{
    cv::Mat mean;
    cv::Mat dev;
    cv::meanStdDev(image, mean, dev);

    if (dev.at<double>(0) <= stddev)
    {
        results.push_back( image );
        image.setTo(mean);
        return;
    }

    const auto width = image.cols;
    const auto height = image.rows;

    split_image(image(cv::Range(0, height / 2), cv::Range(0, width / 2)), stddev, results);
    split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), stddev, results);
    split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), stddev, results);
    split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), stddev, results);
}
}

void merge_image(cv::Mat image, double stddev, std::vector<MergingImg>& mergeIMGs)
{
    bool mergeMore = false;
    for (int i = 0; i < mergeIMGs.size(); ++i)
    {
        int imgMean = mergeIMGs[i].GetMean();
        for (int j = i + 1; j < mergeIMGs.size(); ++j)
        {
            if ((std::abs)(imgMean - mergeIMGs[j].GetMean()) <= stddev)
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


    if (mergeMore)
    {
        auto new_end = std::remove_if(mergeIMGs.begin(), mergeIMGs.end(), [](const MergingImg& img) {return img.GetIsExpired(); });
        mergeIMGs.erase(new_end, mergeIMGs.end());

        merge_image(image, stddev, mergeIMGs);
    }
    else
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
    images splittedImg{};
    split_image(res, stddev, splittedImg);

    // merge part
    std::vector<MergingImg> mergeIMGs;
    for ( int i = 0; i < splittedImg.size(); ++i)
    {
        if (splittedImg[i].size().height != 0
            && splittedImg[i].size().width != 0)
        {
            mergeIMGs.emplace_back(splittedImg, i);
        }
    }
    merge_image(res, stddev, mergeIMGs);
    return res;
}
} // namespace cvlib
