#include "functions.h"
#include "segmentation.h"

#include <chrono>

int main(int argc, char * argv[]) {
    const cv::Mat img = cv::imread("/home/dp/Downloads/ARL_data/19.png");

    std::vector<cv::Rect> segmentation_regions;
    std::vector<float> segmentation_scores;

    std::chrono::high_resolution_clock::time_point t1, t2;
    double duration;

    t1 = std::chrono::high_resolution_clock::now();
    Segmentation::process(img, segmentation_regions, segmentation_scores);
    t2 = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << std::endl << "Segmentation Time: " << duration / 1000 << "s" << std::endl;

    Segmentation::showSegmentationResults(img, segmentation_regions, segmentation_scores);

    cv::waitKey();

    return 0;
}