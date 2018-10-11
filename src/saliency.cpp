#include "functions.h"

#include <iostream>

int main(int argc, char * argv[]) {
    
    cv::namedWindow("Orignal_Img", cv::WINDOW_NORMAL);
    cv::resizeWindow("Orignal_Img", 1200, 1200);
    cv::namedWindow("Salient_Img", cv::WINDOW_NORMAL);
    cv::resizeWindow("Salient_Img", 1200, 1200);

    //const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");
    const cv::Mat img = cv::imread("/home/dp/Downloads/VOCdevkit/VOC2007/JPEGImages/000013.jpg");

    cv::Mat salImg;

    CalculateSaliceny(img, salImg, false);
    
    std::cout << std::endl << salImg.type() << std::endl;
    
    cv::imshow("Orignal_Img", img);
    cv::imshow("Salient_Img", salImg);

    while (cv::waitKey(0) != 'q');

    return 0;
}
