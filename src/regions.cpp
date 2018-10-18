#include "functions.h"

#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>

void graphSegmentation(const cv::Mat & img) {
    DisplayImg(img, "Orignal_Img");

    std::vector<cv::Rect> regions;
    RegionProposalsGraph(img, regions);

    std::cout << std::endl << "Region Proposals - Graph Segmentation: " << regions.size() << std::endl;

    std::vector<cv::Rect> filteredRegions(regions);
    cv::groupRectangles(filteredRegions, 1);
    std::cout << std::endl << "Regions After filter: " << filteredRegions.size() << std::endl;

    DisplayBoundingBoxesInteractive(img, filteredRegions, "Filtered_Regions");
    DisplayBoundingBoxesInteractive(img, regions, "All_Regions");
}

void selectiveSegmentation(const cv::Mat & img) {
    DisplayImg(img, "Orignal_Img");

    std::vector<cv::Rect> regions;
    RegionProposalsSelectiveSearch(img, regions);

    std::cout << std::endl << "Region Proposals - Selective Search Segmentation: " << regions.size() << std::endl;

    std::vector<cv::Rect> filteredRegions(regions);
    cv::groupRectangles(filteredRegions, 1);
    std::cout << std::endl << "Regions After filter: " << filteredRegions.size() << std::endl;

    DisplayBoundingBoxesInteractive(img, filteredRegions, "Filtered_Regions");
    DisplayBoundingBoxesInteractive(img, regions, "All_Regions");
}

void contourSegmentation(const cv::Mat & img) {
    DisplayImg(img, "Orignal_Img");

    std::vector<cv::Rect> regions;
    RegionProposalsContour(img, regions);

    std::cout << std::endl << "Region Proposals - Contour Segmentation: " << regions.size() << std::endl;

    std::vector<cv::Rect> filteredRegions(regions);
    cv::groupRectangles(filteredRegions, 1);
    std::cout << std::endl << "Regions After filter: " << filteredRegions.size() << std::endl;

    DisplayBoundingBoxesInteractive(img, filteredRegions, "Filtered_Regions");
    DisplayBoundingBoxesInteractive(img, regions, "All_Regions");
}

int main(int argc, char * argv[]) {
    const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");
    //const cv::Mat img = cv::imread("/home/dp/Desktop/YOLO/darknet/data/dog.jpg");

    //graphSegmentation(img);
    selectiveSegmentation(img);
    //contourSegmentation(img);

    while (cv::waitKey() != 'q');

    return 0;
}