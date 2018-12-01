#include "functions.h"

#include <iostream>

int main(int argc, char * argv[]) {
    const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");
    //const cv::Mat img = cv::imread("/home/dp/Downloads/VOCdevkit/VOC2007/JPEGImages/000013.jpg");
    DisplayImg(img, "Orignal_Img");

    cv::Mat salImg;
    CalculateSaliceny(img, salImg, false);
    std::cout << std::endl << salImg.type() << std::endl;

    DisplayImg(salImg, "Salient_Img");

    while (cv::waitKey(0) != 'q');

    return 0;
}
