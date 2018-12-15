#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/ximgproc/segmentation.hpp>

#include <opencv2/saliency/saliencyBaseClasses.hpp>
#include <opencv2/saliency/saliencySpecializedClasses.hpp>

void RegionProposalsGraph(const cv::Mat & img, std::vector<cv::Rect> & regions);
void RegionProposalsSelectiveSearch(const cv::Mat & img, std::vector<cv::Rect> & regions, const int resizeHeight = 200);
void RegionProposalsContour(const cv::Mat & img, std::vector<cv::Rect> & regions, const float thresh = 100);

void CalculateSaliceny(const cv::Mat & img, cv::Mat & saliencyMat, bool useFineGrained = true);

void RemoveUnsalient(const cv::Mat & salientImg, const std::vector<cv::Rect> & regions, std::vector<cv::Rect> & highest, const int keepNum = 7);

bool RemoveOverlapping(std::vector<cv::Rect> & regions, float minOveralap = 0.85);

inline char DisplayImg(const cv::Mat & img, const std::string & windowName, int width = 1200, int height = 1200, bool wait = false) {
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, width, height);
    cv::imshow(windowName, img);
    return (wait) ? cv::waitKey() : 0;
}

inline char UpdateImg(const cv::Mat & img, const std::string & windowName, const std::string & windowTitle = "", bool wait = false) {
    cv::imshow(windowName, img);
    if (*windowTitle.c_str())
        cv::setWindowTitle(windowName, windowTitle);
    return (wait) ? cv::waitKey() : 0;
}

//draw single bounding box

inline void DrawBoundingBox(cv::Mat & img, const cv::Rect & rect, const cv::Scalar & color = cv::Scalar(0, 0, 0), bool showCenter = true) {
    cv::rectangle(img, rect, color);
    if (showCenter)
        cv::drawMarker(img, (rect.tl() + rect.br()) / 2, color);
}


//draw multiple bounding boxes

inline void DrawBoundingBoxes(cv::Mat & img, const std::vector<cv::Rect> & regions, const cv::Scalar & color = cv::Scalar(0, 255, 0)) {
    for (const cv::Rect & rect : regions)
        cv::rectangle(img, rect, color);
}

//make sure grid lines fit evenly for optimal behavior

inline void DrawGridLines(cv::Mat & img, int numGrids, const cv::Scalar gridColor = cv::Scalar(255, 255, 255)) {
    const int gridSizeR = img.rows / numGrids;
    const int gridSizeC = img.cols / numGrids;

    for (int r = 0; r < img.rows; r += gridSizeR)
        cv::line(img, cv::Point(r, 0), cv::Point(r, img.cols), gridColor);
    for (int c = 0; c < img.cols; c += gridSizeC)
        cv::line(img, cv::Point(0, c), cv::Point(img.rows, c), gridColor);
}

inline void DisplayBoundingBoxesInteractive(const cv::Mat & img, const std::vector<cv::Rect> regions, const std::string & windowName = "Inteactive_Regions", const int increment = 50) {
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 1200, 1200);
    int numRectsShown = increment; // number of region proposals to show

    while (1) {
        cv::Mat dispImg = img.clone();

        //dipslay only required number of proposals
        for (int i = 0; i < regions.size(); ++i) {
            if (i < numRectsShown)
                cv::rectangle(dispImg, regions[i], cv::Scalar(0, 255, 0));
            else
                break;

        }
        DisplayImg(dispImg, windowName);

        int k = cv::waitKey(); // record key press
        if (k == 'm')
            numRectsShown += increment; // increase total number of rectangles to show by increment
        else if (k == 'l' && numRectsShown > increment)
            numRectsShown -= increment; // decrease total number of rectangles to show by increment
        else if (k == 'c')
            return;
    }
}

inline void viewMostCommonCommon(const std::vector<cv::Rect> & regions, cv::Mat & output) {
    assert(output.type() == CV_16UC1);
    cv::Mat tempSubMat;
    for (const cv::Rect & tempRegion : regions) {
        tempSubMat = output(tempRegion);
        tempSubMat.forEach<uint16_t>([](uint16_t & val, const int *) -> void {
            ++val;
        });
    }
    cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, CV_8UC1);
}

#endif