#include <iostream>
#include <fstream>

#include "functions.h"

//x,y = center[0,1]| w,h = width,height[0,1] - output format of YOLO

void addBoundingBox(const cv::Mat & img, std::vector<cv::Rect> & rects, float x, float y, float w, float h) {
    cv::Point center(x * img.cols, y * img.rows);
    cv::Point offset(img.cols * w / 2, img.rows * h / 2);
    rects.emplace_back(center - offset, center + offset);
}

void drawBoundingBox(cv::Mat & img, const cv::Rect & rect, const cv::Scalar & color = cv::Scalar(0, 0, 0), bool showCenter = true) {
    cv::rectangle(img, rect, color);
    if (showCenter)
        cv::drawMarker(img, (rect.tl() + rect.br()) / 2, color);
}

int main(int argc, char * argv[]) {
    cv::Mat img = cv::imread("/home/dp/Desktop/ARL/darknet-master/networkInput.jpg");

    const int imgSize = 608; //fixed input size of yolo network image
    const int numGrids = 19; //based on layer
    const int gridSize = imgSize / numGrids;
    const std::string fileName = "/home/dp/Desktop/ARL/darknet-master/firstGrid.txt";
    std::vector<cv::Rect> regionProposals;

    std::ifstream of(fileName);
    assert(img.cols == img.rows && img.rows == imgSize);
    assert(of.is_open());



    of.close();

    //draw gridlines
    for (int y = 0; y < imgSize; y += gridSize)
        cv::line(img, cv::Point(0, y), cv::Point(imgSize, y), cv::Scalar(255, 255, 255));
    for (int x = 0; x < imgSize; x += gridSize)
        cv::line(img, cv::Point(x, 0), cv::Point(x, imgSize), cv::Scalar(255, 255, 255));


    float x, y, w, h;

    x = 0.285963148, y = 0.413802862;
    w = 0.235989377, h = 0.124591269;
    addBoundingBox(img, regionProposals, x, y, w, h);
    drawBoundingBox(img, regionProposals.back(), cv::Scalar(0, 0, 255));

    x = 0.288791478, y = 0.424954563;
    w = 0.228967696, h = 0.130756572;
    addBoundingBox(img, regionProposals, x, y, w, h);
    drawBoundingBox(img, regionProposals.back(), cv::Scalar(255, 0, 0));

    x = 0.394665509, y = 0.489420593;
    w = 0.162790462, h = 0.120024256;
    addBoundingBox(img, regionProposals, x, y, w, h);
    drawBoundingBox(img, regionProposals.back(), cv::Scalar(0, 255, 0));

    x = 0.769671261, y = 0.519694567;
    w = 0.236110166, h = 0.273339629;
    addBoundingBox(img, regionProposals, x, y, w, h);
    drawBoundingBox(img, regionProposals.back(), cv::Scalar(0, 0, 255));

    x = 0.767452896, y = 0.547523201;
    w = 0.230168507, h = 0.289458811;
    addBoundingBox(img, regionProposals, x, y, w, h);
    drawBoundingBox(img, regionProposals.back(), cv::Scalar(255, 0, 0));

    x = 0.403326631, y = 0.703275025;
    w = 0.208203584, h = 0.14874813;
    addBoundingBox(img, regionProposals, x, y, w, h);
    drawBoundingBox(img, regionProposals.back(), cv::Scalar(0, 255, 0));


    DisplayImg(img, "Input", 600, 600);
    cv::waitKey();
    return 0;
}