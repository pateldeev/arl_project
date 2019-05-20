#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "vocus2.h"

#include "functions.h"

void trackbarCallback(int pos, void* data) {
    cv::Mat& img = *((cv::Mat*) data);
    cv::Mat disp;
    cv::threshold(img, disp, pos, 255, cv::THRESH_BINARY);
    cv::imshow("Thresholded", disp);
}

int main(int argc, char* argv[]) {
    VOCUS2_Cfg cfg;
    VOCUS2 vocus2(cfg);
    const std::string window_name = "VOCUS2 Saliency Map";

    const cv::Mat img = cv::imread("/home/dp/Downloads/project/data_cumulative/0.png");

    vocus2.process(img);
    cv::Mat sal_map = vocus2.get_salmap();

    CreateThresholdImageWindow(sal_map);

    return 0;
}