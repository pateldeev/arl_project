#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>

#include <opencv2/ximgproc/segmentation.hpp>

#include <opencv2/saliency/saliencyBaseClasses.hpp>
#include <opencv2/saliency/saliencySpecializedClasses.hpp>

cv::Mat GetSubRegionOfMat(const cv::Mat &m, const cv::Rect &r);

char DisplayImg(const cv::Mat &img, const std::string &window_name, int width = 1200, int height = 1200, bool wait = false);
char DisplayImg(const std::string &window_name, const cv::Mat &img, int width = 1200, int height = 1200, bool wait = false);
void DisplayMultipleImages(const std::string &window_name, const std::vector<cv::Mat> &images, unsigned int rows = 2, unsigned int cols = 5, const cv::Size &display_size = cv::Size(1800, 1000), bool wait = false);
char UpdateImg(const cv::Mat &img, const std::string &window_name, const std::string &window_title = "", bool wait = false);

void CreateThresholdImageWindow(const cv::Mat &img, const std::string &window_name_main = "Thresholding");
void ShowHistrogram(const cv::Mat &img, const std::string &window_name = "Histogram");

void DrawBoundingBox(cv::Mat &img, const cv::Rect &rect, const cv::Scalar &color = cv::Scalar(0, 0, 0), bool show_center = false, unsigned int thickness = 1);
void DrawBoundingBoxes(cv::Mat &img, const std::vector<cv::Rect> &regions, const cv::Scalar &color = cv::Scalar(0, 255, 0));
void DisplayBoundingBoxesInteractive(const cv::Mat &img, const std::vector<cv::Rect> &regions, const std::string &window_name = "Inteactive_Regions", const unsigned int increment = 50);

void WriteText(cv::Mat &img, const std::string &text, float font_size = 0.75, const cv::Scalar &font_color = cv::Scalar(0, 0, 255), const cv::Point &pos = cv::Point(10, 25));
void WriteText(cv::Mat &img, const std::vector<std::string> &text, float font_size = 0.75, const cv::Scalar &font_color = cv::Scalar(0, 0, 255), const cv::Point &start_pos = cv::Point(10, 25), const cv::Point &change_per_line = cv::Point(0, 25));

cv::Mat GetGraphSegmentationViewable(const cv::Mat &img_segmented, bool disp_count = false);

void DrawGridLines(cv::Mat &img, int num_grids, const cv::Scalar &grid_color = cv::Scalar(255, 255, 255));

float CalcSpatialEntropy(const cv::Mat &img, const cv::Rect &region, const cv::Rect &region_blackout);
float CalcSpatialEntropy(const cv::Mat &img, const cv::Rect &region);
float CalcSpatialEntropy(const cv::Mat &img);

inline void RegionProposalsGraph(const cv::Mat &segmented_img, std::vector<cv::Rect> &regions) {
    regions.clear();

    double min, max;
    cv::minMaxLoc(segmented_img, &min, &max);
    regions.resize(int(max) + 1);

    for (int r = 0; r < segmented_img.rows; ++r) {
        for (int c = 0; c < segmented_img.cols; ++c) {
            uint32_t segment = segmented_img.at<uint32_t>(r, c);

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

inline void RegionProposalsSelectiveSearch(const cv::Mat &img, std::vector<cv::Rect> &regions, const int resize_h = 200) {
    regions.clear();

    const int old_h = img.rows;
    const int old_w = img.cols;
    const int resize_w = old_w * resize_h / old_h;
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(resize_w, resize_h));

    // speed-up using multithreads
    //cv::setUseOptimized(true);
    //cv::setNumThreads(8);

    // create Selective Search Segmentation Object using default parameters
    cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();

    ss->setBaseImage(img_resized); // set input image on which we will run segmentation

    ss->switchToSelectiveSearchQuality(); // change to high quality

    //run selective search segmentation on input image
    std::vector<cv::Rect> regions_resized;

    std::chrono::high_resolution_clock::time_point t1, t2;
    t1 = std::chrono::high_resolution_clock::now();

    ss->process(regions_resized);

    t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    // rescale rectangles back for original image
    for (const cv::Rect & rec : regions_resized) {
        int x = old_w * ((double) rec.x / resize_w);
        int y = old_h * ((double) rec.y / resize_h);
        int width = ((double) rec.width / resize_w) * old_w;
        int height = ((double) rec.height / resize_h) * old_h;

        regions.push_back(cv::Rect(x, y, width, height));
    }

    std::cout << std::endl << "Region Proposals - Selective Search Segmentation: " << regions.size() << std::endl << "Time: " << duration;
}

inline void RegionProposalsContour(const cv::Mat &img, std::vector<cv::Rect> &regions, const float thresh = 100) {
    regions.clear();

    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::blur(img_gray, img_gray, cv::Size(3, 3));

    cv::Mat thresh_output;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::threshold(img_gray, thresh_output, thresh, 255, cv::THRESH_BINARY);
    cv::findContours(thresh_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    std::vector<std::vector < cv::Point >> contours_poly(contours.size());
    regions.resize(contours.size());

    for (unsigned int i = 0; i < contours.size(); ++i) {
        cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
        regions[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
    }
}

inline void CalculateSaliceny(const cv::Mat &img, cv::Mat &saliency_map, bool use_fine_grained = true) {
    if (use_fine_grained) {
        cv::Ptr<cv::saliency::StaticSaliencyFineGrained> saliency = cv::saliency::StaticSaliencyFineGrained::create();
        saliency->computeSaliency(img, saliency_map);
    } else {
        cv::Ptr<cv::saliency::StaticSaliencySpectralResidual> saliency = cv::saliency::StaticSaliencySpectralResidual::create();
        saliency->computeSaliency(img, saliency_map);
        cv::normalize(saliency_map, saliency_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    }
}

inline float AvgSaliency(const cv::Mat &saliency_map, const cv::Rect & region) {
    int sum = 0, count = 0;
    for (int row = region.y; row < region.y + region.height; ++row)
        for (int col = region.x; col < region.x + region.width; ++col, ++count)
            sum += saliency_map.at<uint8_t>(row, col);
    return ((float) sum / count);
}

inline float NumSaliency(const cv::Mat &saliency_map, const cv::Rect & region) {
    int salient = 0, total = 0;
    const int thresh = 120;
    for (int row = region.y; row < region.y + region.height; ++row) {
        for (int col = region.x; col < region.x + region.width; ++col) {
            ++total;
            if (saliency_map.at<uint8_t>(row, col) > thresh)
                ++salient;
        }
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

inline void RemoveUnsalient(const cv::Mat &saliency_map, const std::vector<cv::Rect> &regions, std::vector<cv::Rect> &highest, const int keep_num = 7) {
    CV_Assert(regions.size() >= keep_num);
    CV_Assert(saliency_map.type() == CV_8UC1);
    highest.resize(keep_num);

    std::multiset<region> orderedRegions;

    for (unsigned int i = 0; i < regions.size(); ++i)
        orderedRegions.insert(region(i, NumSaliency(saliency_map, regions[i])));

    std::multiset<region>::reverse_iterator rit = orderedRegions.rbegin();
    for (unsigned int i = 0; i < keep_num; ++i, ++rit)
        highest[i] = regions[rit->m_index];
}

inline void RemoveOverlapping(std::vector<cv::Rect> &regions, float min_overlap = 0.85) {
    unsigned int size = regions.size();
    cv::Rect overlap;
    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = i + 1; j < size; ++j) {
            if (!regions[i].empty() && !regions[j].empty()) {
                overlap = regions[i] & regions[j];
                if (overlap.area() >= min_overlap * regions[i].area() && overlap.area() >= min_overlap * regions[j].area()) {
                    regions.push_back(regions[i] | regions[j]);
                    regions[i] = regions[j] = cv::Rect();
                }
            }
        }
    }

    size = regions.size();
    regions.erase(std::remove_if(regions.begin(), regions.end(),
            [](const cv::Rect & rect)->bool {
                return rect.empty(); }), regions.end());

    if (size != regions.size())
        RemoveOverlapping(regions, min_overlap);
}


// Counts number of regions that contain each pixel value
// heatMap should be a CV_16UC1 image of all zeros of the correct size 

inline void CreateHeatMap(const std::vector<cv::Rect> &regions, cv::Mat & heat_map) {
    CV_Assert(heat_map.type() == CV_16UC1);

    for (const cv::Rect & r : regions)
        for (int y = r.tl().y; y < r.br().y; ++y)
            for (int x = r.tl().x; x < r.br().x; ++x)
                ++heat_map.at<uint16_t>(y, x);
}
#endif