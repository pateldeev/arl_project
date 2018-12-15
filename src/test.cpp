#include "functions.h"
#include "selectiveSegmentation.h"

#include <iostream>
#include <chrono>

int main(int argc, char * argv[]) {
    //const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");
    const cv::Mat img = cv::imread("/home/dp/Desktop/ARL/darknet-master/networkInput.jpg");
    std::vector<cv::Rect> proposals;

    //rescale image
    const int resizeHeight = 200;
    const int oldHeight = img.rows;
    const int oldWidth = img.cols;
    const int resizeWidth = oldWidth * resizeHeight / oldHeight;
    cv::Mat imgResized;
    cv::resize(img, imgResized, cv::Size(resizeWidth, resizeHeight));
    DisplayImg(imgResized, "Resized");

    std::chrono::high_resolution_clock::time_point t1, t2;
    t1 = std::chrono::high_resolution_clock::now();

    Segmentation::process(imgResized, proposals, 150, 150, 0.8);

    t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    //rescale rectangles back for original image
    for (cv::Rect & rec : proposals) {
        int x = oldWidth * (double(rec.x) / resizeWidth);
        int y = oldHeight * (double(rec.y) / resizeHeight);
        int width = oldWidth * (double(rec.width) / resizeWidth);
        int height = oldHeight * (double(rec.height) / resizeHeight);

        rec = cv::Rect(x, y, width, height);
    }

    std::cout << std::endl << "Region Proposals: " << proposals.size() << std::endl << "Time: " << duration << std::endl;

    DisplayBoundingBoxesInteractive(img, proposals, "Proposals");


    return 0;
}

int main2(int argc, char * argv[]) {
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