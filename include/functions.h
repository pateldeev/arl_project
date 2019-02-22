#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/ximgproc/segmentation.hpp>

#include <opencv2/saliency/saliencyBaseClasses.hpp>
#include <opencv2/saliency/saliencySpecializedClasses.hpp>

void RegionProposalsGraph(const cv::Mat &segmentedImg, std::vector<cv::Rect> &regions);
void RegionProposalsSelectiveSearch(const cv::Mat &img, std::vector<cv::Rect> &regions, const int resizeHeight = 200);
void RegionProposalsContour(const cv::Mat &img, std::vector<cv::Rect> &regions, const float thresh = 100);

void CalculateSaliceny(const cv::Mat &img, cv::Mat &saliencyMat, bool useFineGrained = true);

void RemoveUnsalient(const cv::Mat &salientImg, const std::vector<cv::Rect> &regions, std::vector<cv::Rect> &highest, const int keepNum = 7);

void RemoveOverlapping(std::vector<cv::Rect> &regions, float minOveralap = 0.85);

// Counts number of regions that contain each pixel value
// heatMap should be a CV_16UC1 image of all zeros of the correct size 
void CreateHeatMap(const std::vector<cv::Rect> &regions, cv::Mat &heatMap);

inline char DisplayImg(const cv::Mat &img, const std::string &windowName, int width = 1200, int height = 1200, bool wait = false) {
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, width, height);
    cv::imshow(windowName, img);
    return (wait) ? cv::waitKey() : 0;
}

inline char DisplayImg(const std::string &windowName, const cv::Mat &img, int width = 1200, int height = 1200, bool wait = false) {
    return DisplayImg(img, windowName, width, height, wait);
}

inline char UpdateImg(const cv::Mat &img, const std::string &windowName, const std::string &windowTitle = "", bool wait = false) {
    cv::imshow(windowName, img);
    if (*windowTitle.c_str())
        cv::setWindowTitle(windowName, windowTitle);
    return (wait) ? cv::waitKey() : 0;
}

//draw single bounding box

inline void DrawBoundingBox(cv::Mat &img, const cv::Rect &rect, const cv::Scalar &color = cv::Scalar(0, 0, 0), bool showCenter = false) {
    cv::rectangle(img, rect, color);
    if (showCenter)
        cv::drawMarker(img, (rect.tl() + rect.br()) / 2, color);
}


//draw multiple bounding boxes

inline void DrawBoundingBoxes(cv::Mat &img, const std::vector<cv::Rect> &regions, const cv::Scalar &color = cv::Scalar(0, 255, 0)) {
    for (const cv::Rect & rect : regions)
        cv::rectangle(img, rect, color);
}

//make sure grid lines fit evenly for optimal behavior

inline void DrawGridLines(cv::Mat &img, int numGrids, const cv::Scalar &gridColor = cv::Scalar(255, 255, 255)) {
    const int gridSizeR = img.rows / numGrids;
    const int gridSizeC = img.cols / numGrids;

    for (int r = 0; r < img.rows; r += gridSizeR)
        cv::line(img, cv::Point(r, 0), cv::Point(r, img.cols), gridColor);
    for (int c = 0; c < img.cols; c += gridSizeC)
        cv::line(img, cv::Point(0, c), cv::Point(img.rows, c), gridColor);
}

//make sure grid lines fit evenly for optimal behavior

inline void DisplayBoundingBoxesInteractive(const cv::Mat &img, const std::vector<cv::Rect> &regions, const std::string &windowName = "Inteactive_Regions", const unsigned int increment = 50) {
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 1200, 1200);
    unsigned int numRectsShown = increment; // number of region proposals to show

    while (1) {
        cv::Mat dispImg = img.clone();

        //display only required number of proposals
        for (unsigned int i = 0; i < regions.size(); ++i) {
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

inline void WriteText(cv::Mat &img, const std::string &text, float font_size = 0.75, const cv::Scalar &font_color = cv::Scalar(0, 0, 255), const cv::Point &pos = cv::Point(10, 25)) {
    cv::putText(img, text, pos, cv::FONT_HERSHEY_DUPLEX, font_size, font_color, 1.25);
}

inline void WriteText(cv::Mat &img, const std::vector<std::string> &text, float font_size = 0.75, const cv::Scalar &font_color = cv::Scalar(0, 0, 255), const cv::Point &start_pos = cv::Point(10, 25), const cv::Point &change_per_line = cv::Point(0, 25)) {
    cv::Point pos = start_pos;
    for (const std::string &t : text) {
        WriteText(img, t, font_size, font_color, pos);
        pos += change_per_line;
    }
}

//show multiple CV_8UC1 or CV_8UC3 images in same window - doesn't check number of images provided to make sure it fits

inline void ShowManyImages(const std::string &windowName, const std::vector<cv::Mat> &images, int numRows = 2, int numColumns = 5) {
    const int dispW = 1800, dispH = 950; //size of collective image
    cv::Mat dispImg = cv::Mat::zeros(dispH, dispW, CV_8UC3);
    const int imgTargetW = dispW / numColumns, imgTargetH = dispH / numRows; //target size of each small subimage

    int tempW = 0, tempH = 0; //used to keep track of top left corner of each subimage
    cv::Mat imgResized;

    for (const cv::Mat & img : images) {
        CV_Assert(img.type() == CV_8UC1 || img.type() == CV_8UC3);

        int imgResizeByHeightW = img.cols * (float(imgTargetH) / img.rows);
        int imgResizeByWidthH = img.rows * (float(imgTargetW) / img.cols);

        if (imgResizeByHeightW <= imgTargetW) //rescale so height fits perfectly
            cv::resize(img, imgResized, cv::Size(imgResizeByHeightW, imgTargetH));
        else //rescale so width fits perfectly
            cv::resize(img, imgResized, cv::Size(imgTargetW, imgResizeByWidthH));

        //copy over relevant pixel values
        for (unsigned int r = 0; r < imgResized.rows; ++r)
            for (unsigned int c = 0; c < imgResized.cols; ++c) {
                if (img.type() == CV_8UC3)
                    dispImg.at<cv::Vec3b>(r + tempH, c + tempW) = imgResized.at<cv::Vec3b>(r, c);
                else
                    dispImg.at<cv::Vec3b>(r + tempH, c + tempW) = cv::Vec3b(imgResized.at<uint8_t>(r, c), imgResized.at<uint8_t>(r, c), imgResized.at<uint8_t>(r, c));
            }

        //calculate start position of next image
        tempW += imgTargetW;
        if (tempW >= dispW) {
            tempW = 0;
            tempH += imgTargetH;
        }
    }

    DisplayImg(dispImg, windowName, dispW, dispH);
}

//returns viewable representation of graph segmentation

inline cv::Mat GetGraphSegmentationViewable(const cv::Mat &imgSegmented, bool disp_count = false) {
    CV_Assert(imgSegmented.type() == CV_32S);

    double min, max;
    cv::minMaxLoc(imgSegmented, &min, &max);

    cv::Mat disp_img = cv::Mat::zeros(imgSegmented.rows, imgSegmented.cols, CV_8UC3);

    const uint32_t * p;
    uint8_t * p2;

    //helper function to determine color
    auto color_mapping = [](int segment_id) -> cv::Scalar {
        double base = double(segment_id) * 0.618033988749895 + 0.24443434;
        cv::Scalar c(std::fmod(base, 1.2), 0.95, 0.80);
        cv::Mat in(1, 1, CV_32FC3), out(1, 1, CV_32FC3);

        float * p = in.ptr<float>(0);
        p[0] = float(c[0]) * 360.0f;
        p[1] = float(c[1]);
        p[2] = float(c[2]);

        cv::cvtColor(in, out, cv::COLOR_HSV2RGB);
        cv::Scalar t;
        cv::Vec3f p2 = out.at<cv::Vec3f>(0, 0);

        t[0] = int(p2[0] * 255);
        t[1] = int(p2[1] * 255);
        t[2] = int(p2[2] * 255);

        return t;
    };

    for (int i = 0; i < imgSegmented.rows; ++i) {
        p = imgSegmented.ptr<uint32_t>(i);
        p2 = disp_img.ptr<uint8_t>(i);

        for (int j = 0; j < imgSegmented.cols; ++j) {
            cv::Scalar color = color_mapping(int(p[j]));
            p2[j * 3] = (uint8_t) color[0];
            p2[j * 3 + 1] = (uint8_t) color[1];
            p2[j * 3 + 2] = (uint8_t) color[2];
        }
    }

    if (disp_count)
        WriteText(disp_img, std::to_string(int(max + 1)));

    return disp_img;
}

#if 0

inline void viewMostCommonCommon(const std::vector<cv::Rect> &regions, cv::Mat &output) {
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

#endif