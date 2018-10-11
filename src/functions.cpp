#include "functions.h"

#include <set>

#include <iostream>

void CalculateRegionProposals(const cv::Mat & img, std::vector<cv::Rect> & regions, const int resizeHeight) {
    regions.clear();

    const int oldHeight = img.rows;
    const int oldWidth = img.cols;
    const int resizeWidth = oldWidth * resizeHeight / oldHeight;
    cv::Mat imgResized;
    cv::resize(img, imgResized, cv::Size(resizeWidth, resizeHeight));

    // speed-up using multithreads
    cv::setUseOptimized(true);
    cv::setNumThreads(4);

    // create Selective Search Segmentation Object using default parameters
    cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();

    ss->setBaseImage(imgResized); // set input image on which we will run segmentation

    ss->switchToSelectiveSearchQuality(); // change to high quality

    // run selective search segmentation on input image
    std::vector<cv::Rect> regionsResized;
    ss->process(regionsResized);

    // rescale rectangles back for original image
    for (const cv::Rect & rec : regionsResized) {
        int x = oldWidth * ((double) rec.x / resizeWidth);
        int y = oldHeight * ((double) rec.y / resizeHeight);
        int width = ((double) rec.width / resizeWidth) * oldWidth;
        int height = ((double) rec.height / resizeHeight) * oldHeight;

        regions.push_back(cv::Rect(x, y, width, height));
    }
}

void CalculateSaliceny(const cv::Mat & img, cv::Mat & saliencyMat, bool useFineGrained) {
    if (useFineGrained) {
        cv::Ptr<cv::saliency::StaticSaliencyFineGrained> saliency = cv::saliency::StaticSaliencyFineGrained::create();
        saliency->computeSaliency(img, saliencyMat);
    } else {
        cv::Ptr<cv::saliency::StaticSaliencySpectralResidual> saliency = cv::saliency::StaticSaliencySpectralResidual::create();
        saliency->computeSaliency(img, saliencyMat);
        cv::normalize(saliencyMat, saliencyMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    }
}

float avgSaliency(const cv::Mat & salientImg, const cv::Rect & region) {
    int sum = 0, count = 0;
    for (int row = region.y; row < region.y + region.height; ++row)
        for (int col = region.x; col < region.x + region.width; ++col) {
            ++count;
            sum += salientImg.at<uint8_t>(row,col);
        }

    return ((float) sum / count);
}

struct region {
    unsigned int m_index;
    float m_data;

    region(unsigned int index, float data) : m_index(index), m_data(data) {
    }

    bool operator<(const region & other) const {
        return m_data < other.m_data;
    }
};

void RemoveUnsalient(const cv::Mat & salientImg, const std::vector<cv::Rect> & regions, std::vector<cv::Rect> & highest, const int keepNum) {
    assert(regions.size() >= keepNum);
    assert(salientImg.type() == CV_8UC1);
    highest.clear();

    std::multiset<region> orderedRegions;

    for (int i = 0; i < regions.size(); ++i) {
        orderedRegions.insert(region(i, avgSaliency(salientImg, regions[i])));
    }

    std::multiset<region>::reverse_iterator rit = orderedRegions.rbegin();
    for (int i = 1; i <= keepNum; ++i, ++rit) {
        highest.push_back(regions[rit->m_index]);
    }

}


