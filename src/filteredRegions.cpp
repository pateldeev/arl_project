#include "functions.h"

#include <iostream>

#include <opencv2/objdetect/objdetect.hpp>

int main(int argc, char * argv[]) {

    const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");
    //const cv::Mat img = cv::imread("/home/dp/Desktop/sample/other_sequences/rearrange/0.png");
    //const cv::Mat img = cv::imread("/home/dp/Desktop/YOLO/darknet/data/dog.jpg");
    DisplayImg(img, "Orignal_Img");

    cv::Mat salImg;
    salImg = cv::imread("/home/dp/Desktop/sam-master/predictions/001.jpg", cv::IMREAD_GRAYSCALE);
    //salImg = cv::imread("/home/dp/Desktop/sam-master/predictions/0.png", cv::IMREAD_GRAYSCALE);
    //salImg = cv::imread("/home/dp/Desktop/sam-master/predictions/dog.jpg", cv::IMREAD_GRAYSCALE);
    //CalculateSaliceny(img, salImg, true);
    DisplayImg(salImg, "Salient_Img");

    std::vector<cv::Rect> regions;
    RegionProposalsSelectiveSearch(img, regions);
    const int keep = regions.size() / 20;
    std::cout << std::endl << "Keeping top " << keep << "/" << regions.size() << " region proposals" << std::endl;

    std::vector<cv::Rect> keptRegions;
    RemoveUnsalient(salImg, regions, keptRegions, keep);

    std::cout << std::endl << keptRegions.size() << std::endl;

    cv::Mat dispImg = img.clone();
    for (const cv::Rect & rect : keptRegions) {
        cv::rectangle(dispImg, rect, cv::Scalar(0, 255, 0));
    }

    DisplayImg(dispImg, "Salient_Regions");

    cv::groupRectangles(keptRegions, 1, 0.25);

    std::cout << std::endl << keptRegions.size() << std::endl;

    dispImg = img.clone();
    for (const cv::Rect & rect : keptRegions) {
        cv::rectangle(dispImg, rect, cv::Scalar(0, 255, 0));
    }
    DisplayImg(dispImg, "Filtered_Regions");

#if 0
    RemoveOverlapping(img, keptRegions, 0.2);
    dispImg = img.clone();
    for (const cv::Rect & rect : keptRegions) {
        cv::rectangle(dispImg, rect, cv::Scalar(0, 255, 0));
    }
    DisplayImg(dispImg, "Test");
#endif
    
    std::cout << std::endl << "DONE: PRESS q" << std::endl;

    while (cv::waitKey(0) != 'q');

    return 0;
}