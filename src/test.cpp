#include "functions.h"

#include <iostream>

int main(int argc, char * argv[]) {
    const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");
    cv::Mat regionImg = img.clone();

    std::vector<cv::Rect> regions;
    regions.push_back(cv::Rect(cv::Point(100, 100), cv::Point(500, 500)));
    regions.push_back(cv::Rect(cv::Point(700, 700), cv::Point(1000, 1000)));
    regions.push_back(cv::Rect(cv::Point(50, 150), cv::Point(950, 1025)));

    DrawBoundingBoxes(regionImg, regions);
    DisplayImg(regionImg, "regions");

    regionImg = img.clone();
    RemoveOverlapping(regions);
    DrawBoundingBoxes(regionImg, regions);
    DisplayImg(regionImg, "regions_all");

    while (cv::waitKey(0) != 'q');

    return 0;
}