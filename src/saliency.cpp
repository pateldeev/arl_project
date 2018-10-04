#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/saliency/saliencyBaseClasses.hpp>
#include <opencv2/saliency/saliencySpecializedClasses.hpp>

#include <iostream>

int main(int argc, char * argv[]) {

    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::resizeWindow("Image", 1200, 1200);
    cv::namedWindow("Salient_Img", cv::WINDOW_NORMAL);
    cv::resizeWindow("Salient_Img", 1200, 1200);

    const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");

    cv::Mat salImg;

    cv::Ptr<cv::saliency::StaticSaliencyFineGrained> saliency = cv::saliency::StaticSaliencyFineGrained::create();
    saliency->computeSaliency(img, salImg);

    cv::imshow("Image", img);
    cv::imshow("Salient_Img", salImg);
 
    while (cv::waitKey(0) != 'q');

    return 0;
}
