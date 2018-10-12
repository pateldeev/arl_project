#include "functions.h"

#include <set>

#include <iostream>

void RegionProposalsGraph(const cv::Mat & img, std::vector<cv::Rect> & regions) {
    regions.clear();

    cv::Ptr<cv::ximgproc::segmentation::GraphSegmentation> ss = cv::ximgproc::segmentation::createGraphSegmentation();
    cv::Mat output;
    ss->processImage(img, output);

    double min, max;
    cv::minMaxLoc(output, &min, &max);
    int numSegments = (int) max + 1;

    regions.resize(numSegments);

    for (int r = 0; r < output.rows; r++) {
        for (int c = 0; c < output.cols; c++) {
            uint segment = output.at<uint>(r, c);
            if (!regions[segment].contains(cv::Point(r, c))) {

                if (regions[segment].tl() == regions[segment].br() && regions[segment].tl().x == 0 && regions[segment].tl().y == 0) {
                    regions[segment] = cv::Rect(cv::Point(c, r), cv::Point(c, r));
                } else {
                    if (c < regions[segment].tl().x) {
                        if (r < regions[segment].tl().y)
                            regions[segment] = cv::Rect(cv::Point(c, r), regions[segment].br());
                        else if (r < regions[segment].br().y)
                            regions[segment] = cv::Rect(cv::Point(c, regions[segment].tl().y), regions[segment].br());
                        else
                            regions[segment] = cv::Rect(cv::Point(c, regions[segment].tl().y), cv::Point(regions[segment].br().x, r));
                    } else if (c < regions[segment].br().x) {
                        if (r < regions[segment].tl().y)
                            regions[segment] = cv::Rect(cv::Point(regions[segment].tl().x, r), regions[segment].br());
                        else if (r > regions[segment].y)
                            regions[segment] = cv::Rect(regions[segment].tl(), cv::Point(regions[segment].br().x, r));
                    } else {
                        if (r < regions[segment].tl().y)
                            regions[segment] = cv::Rect(cv::Point(regions[segment].tl().x, r), cv::Point(c, regions[segment].br().y));
                        else if (r < regions[segment].br().y)
                            regions[segment] = cv::Rect(regions[segment].tl(), cv::Point(c, regions[segment].br().y));
                        else
                            regions[segment] = cv::Rect(regions[segment].tl(), cv::Point(c, r));
                    }
                }
            }
        }
    }
}

void RegionProposalsSelectiveSearch(const cv::Mat & img, std::vector<cv::Rect> & regions, const int resizeHeight) {
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

void RegionProposalsContour(const cv::Mat & img, std::vector<cv::Rect> & regions, const float thresh) {
    regions.clear();

    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::blur(img_gray, img_gray, cv::Size(3, 3));

    cv::Mat thresh_output;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::threshold(img_gray, thresh_output, thresh, 255, cv::THRESH_BINARY);
    cv::findContours(thresh_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    std::vector<std::vector<cv::Point>> contours_poly(contours.size());
    regions.resize(contours.size());

    for (size_t i = 0; i < contours.size(); ++i) {
        cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
        regions[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
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
            sum += salientImg.at<uint8_t>(row, col);
        }

    return ((float) sum / count);
}

float numSaliency(const cv::Mat & salientImg, const cv::Rect & region) {
    int salient = 0, total = 0;
    const int thresh = 120;
    for (int row = region.y; row < region.y + region.height; ++row)
        for (int col = region.x; col < region.x + region.width; ++col) {
            ++total;
            if (salientImg.at<uint8_t>(row, col) > thresh)
                ++salient;
        }

    return ((float) salient / total);
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
        orderedRegions.insert(region(i, numSaliency(salientImg, regions[i])));
    }

    std::multiset<region>::reverse_iterator rit = orderedRegions.rbegin();
    for (int i = 1; i <= keepNum; ++i, ++rit) {
        highest.push_back(regions[rit->m_index]);
    }
}

void RemoveOverlapping(const cv::Mat & img, std::vector<cv::Rect> & regions, float minOverlap) {
    int numCombined = 1;
    cv::Rect overlap;
    std::vector<bool> removeRegion(regions.size(), false);
    std::vector<cv::Rect> combinedRegions;

    cv::Mat dispImg = img.clone();
    cv::rectangle(dispImg, regions[0], cv::Scalar(0, 255, 0));
    cv::rectangle(dispImg, regions[1], cv::Scalar(0, 255, 0));
    DisplayImg(dispImg, "asdf");
    while (cv::waitKey(0) != 'c');

    while (numCombined) {
        numCombined = 0;

        std::cout << std::endl << regions.size() << std::endl;

        for (int i = 0; i < regions.size(); ++i) {
            for (int j = i + 1; j < regions.size(); ++j) {
                if (!removeRegion[j]) {
                    overlap = regions[i] & regions[j];

                    if (overlap.area() >= minOverlap * std::min(regions[i].area(), regions[j].area())) {
                        std::cout << std::endl << "HERE : " << overlap.area() << std::endl;

                        dispImg = img.clone();
                        cv::rectangle(dispImg, regions[i], cv::Scalar(0, 255, 0));
                        cv::rectangle(dispImg, regions[j], cv::Scalar(0, 255, 0));
                        cv::rectangle(dispImg, (regions[i] | regions[j]), cv::Scalar(0, 0, 255));
                        DisplayImg(dispImg, "asdf2");

                        while (cv::waitKey(0) != 'c');

                        removeRegion[i] = true;
                        removeRegion[j] = true;
                        regions.push_back((regions[i] | regions[j]));
                        removeRegion.push_back(false);
                        j = regions.size();
                        ++numCombined;
                    }
                    std::cout << std::endl << i << "|" << j << "-" << regions.size() << std::endl;
                    while (cv::waitKey(0) != 'c');

                }
            }
        }

        assert(removeRegion.size() == regions.size());

        for (int i = 0; i < removeRegion.size(); ++i) {
            if (!removeRegion[i]) {
                combinedRegions.push_back(regions[i]);
            }
        }

        regions = combinedRegions;
        combinedRegions.clear();
        std::cout << std::endl << "|" << numCombined << "|" << regions.size() << std::endl;
    }
}

#if 0

cv::Scalar hsv_to_rgb(cv::Scalar c) {
    cv::Mat in(1, 1, CV_32FC3);
    cv::Mat out(1, 1, CV_32FC3);

    float * p = in.ptr<float>(0);

    p[0] = (float) c[0] * 360.0f;
    p[1] = (float) c[1];
    p[2] = (float) c[2];

    cvtColor(in, out, cv::COLOR_HSV2RGB);

    cv::Scalar t;

    cv::Vec3f p2 = out.at<cv::Vec3f>(0, 0);

    t[0] = (int) (p2[0] * 255);
    t[1] = (int) (p2[1] * 255);
    t[2] = (int) (p2[2] * 255);

    return t;
}

cv::Scalar color_mapping(int segment_id) {

    double base = (double) (segment_id) * 0.618033988749895 + 0.24443434;

    return hsv_to_rgb(cv::Scalar(fmod(base, 1.2), 0.95, 0.80));

}

void ShowGraphSegmentation(const cv::Mat & img) {
    cv::Mat output, outputImg;
    cv::Ptr<cv::ximgproc::segmentation::GraphSegmentation> ss = cv::ximgproc::segmentation::createGraphSegmentation();
    ss->processImage(img, output);

    double min, max;
    cv::minMaxLoc(output, &min, &max);

    int nb_segs = (int) max + 1;

    std::cout << nb_segs << " segments" << std::endl;

    outputImg = cv::Mat::zeros(output.rows, output.cols, CV_8UC3);

    uint* p;
    uchar* p2;

    for (int i = 0; i < output.rows; i++) {

        p = output.ptr<uint>(i);
        p2 = outputImg.ptr<uchar>(i);

        for (int j = 0; j < output.cols; j++) {
            cv::Scalar color = color_mapping(p[j]);
            p2[j * 3] = (uchar) color[0];
            p2[j * 3 + 1] = (uchar) color[1];
            p2[j * 3 + 2] = (uchar) color[2];
        }
    }

    DisplayImg(img, "img");
    DisplayImg(outputImg, "output");
}
#endif