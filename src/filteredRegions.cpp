#include "functions.h"

#include <iostream>

int main(int argc, char * argv[]) {

    const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");
    //const cv::Mat img = cv::imread("/home/dp/Downloads/VOCdevkit/VOC2007/JPEGImages/000013.jpg");
    cv::namedWindow("Orignal_Img", cv::WINDOW_NORMAL);
    cv::resizeWindow("Orignal_Img", 1200, 1200);
    cv::imshow("Orignal_Img", img);

    cv::Mat salImg;
    salImg = cv::imread("/home/dp/Desktop/sam-master/predictions/001.jpg", cv::IMREAD_GRAYSCALE);
    //CalculateSaliceny(img, salImg, false);
    cv::namedWindow("Salient_Img", cv::WINDOW_NORMAL);
    cv::resizeWindow("Salient_Img", 1200, 1200);
    cv::imshow("Salient_Img", salImg);

    const int keep = 20;

    std::vector<cv::Rect> regions;
    CalculateRegionProposals(img, regions);
    std::cout << std::endl << "Keeping top " << keep << "/" << regions.size() << " region proposals" << std::endl;

    std::vector<cv::Rect> keptRegions;
    RemoveUnsalient(salImg, regions, keptRegions, keep);

    std::cout << std::endl << keptRegions.size() << std::endl;

    cv::Mat dispImg = img.clone();

    for (const cv::Rect & rect : keptRegions) {
        cv::rectangle(dispImg, rect, cv::Scalar(0, 255, 0));
    }

    cv::namedWindow("Salient_Regions", cv::WINDOW_NORMAL);
    cv::resizeWindow("Salient_Regions", 1200, 1200);
    cv::imshow("Salient_Regions", dispImg);


    while (cv::waitKey(0) != 'q');

    return 0;
}