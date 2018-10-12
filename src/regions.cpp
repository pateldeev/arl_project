#include "functions.h"

#include <iostream>

void graphSegmentation(const cv::Mat & img) {
    DisplayImg(img, "Orignal_Img");

    std::vector<cv::Rect> regions;
    RegionProposalsGraph(img, regions);

    std::cout << std::endl << "Region Proposals - Graph Segmentation: " << regions.size() << std::endl;

    DisplayBoundingBoxesInteractive(img, regions);
}

void selectiveSegmentation(const cv::Mat & img) {
    DisplayImg(img, "Orignal_Img");

    std::vector<cv::Rect> regions;
    RegionProposalsSelectiveSearch(img, regions);

    std::cout << std::endl << "Region Proposals - Selective Search Segmentation: " << regions.size() << std::endl;

    DisplayBoundingBoxesInteractive(img, regions);
}

void contourSegmentation(const cv::Mat & img) {
    DisplayImg(img, "Orignal_Img");

    std::vector<cv::Rect> regions;
    RegionProposalsContour(img, regions);
    
    std::cout << std::endl << "Region Proposals - Contour Segmentation: " << regions.size() << std::endl;

    DisplayBoundingBoxesInteractive(img, regions);
}

int main(int argc, char * argv[]) {
    const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");

    //graphSegmentation(img);
    selectiveSegmentation(img);
    //contourSegmentation(img);

    return 0;
}