#include "functions.h"

#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>

void graphSegmentation(const cv::Mat & img) {
    DisplayImg(img, "Orignal_Img");

    cv::Ptr<cv::ximgproc::segmentation::GraphSegmentation> ss = cv::ximgproc::segmentation::createGraphSegmentation();
    cv::Mat output;
    ss->processImage(img, output);
    DisplayImg(GetGraphSegmentationViewable(output), "Segmentation_Graph");

    std::vector<cv::Rect> regions;
    RegionProposalsGraph(output, regions);

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

    DisplayBoundingBoxesInteractive(img, regions, "All_Regions");
    return;

    std::vector<cv::Rect> filteredRegions(regions);
    //cv::groupRectangles(filteredRegions, 1);
    RemoveOverlapping(filteredRegions, 0.7);
    std::cout << std::endl << "Regions After filter: " << filteredRegions.size() << std::endl;

#if 0
    cv::Mat mostCommon(img.rows, img.cols, CV_16UC1, cv::Scalar(0));
    viewMostCommonCommon(regions, mostCommon);
    DisplayImg(mostCommon, "Most common");
    //mostCommon = cv::Mat(img.rows, img.cols, CV_16UC1, cv::Scalar(0));
    //viewMostCommonCommon(filteredRegions, mostCommon);
    //DisplayImg(mostCommon, "Most common filtered");
    std::vector<cv::Rect> keptRegions;
    RemoveUnsalient(mostCommon, regions, keptRegions, regions.size() / 10);
    std::cout << std::endl << "Salient Kept: " << keptRegions.size() << std::endl;
    cv::Mat dispImg = img.clone();
    DrawBoundingBoxes(dispImg, keptRegions);
    DisplayImg(dispImg, "Salient_Regions");

    cv::groupRectangles(keptRegions, 1, 0.25);
    std::cout << std::endl << keptRegions.size() << std::endl;
    dispImg = img.clone();
    DrawBoundingBoxes(dispImg, keptRegions);
    DisplayImg(dispImg, "Filtered_Regions");
#else
    cv::Mat imgCpy = img.clone();
    cv::GaussianBlur(imgCpy, imgCpy, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    cv::Mat salImg(imgCpy.rows, imgCpy.cols, CV_16SC1);
    cv::Laplacian(imgCpy, salImg, CV_16S);
    std::vector<cv::Mat> channels(3);
    cv::split(salImg, channels);
    salImg = channels[0];
    cv::convertScaleAbs(salImg, salImg);
    cv::normalize(salImg, salImg, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    DisplayImg(salImg, "Laplacian");

    std::vector<cv::Rect> keptRegions;
    RemoveUnsalient(salImg, regions, keptRegions, regions.size() / 10);
    std::cout << std::endl << "Salient Kept: " << keptRegions.size() << std::endl;
    cv::Mat dispImg = img.clone();
    //DrawBoundingBoxes(dispImg, keptRegions);
    //DisplayImg(dispImg, "Salient_Regions");

    cv::groupRectangles(keptRegions, 1, 0.25);
    std::cout << std::endl << keptRegions.size() << std::endl;
    dispImg = img.clone();
    //DrawBoundingBoxes(dispImg, keptRegions);
    //DisplayImg(dispImg, "Filtered_Regions");

#endif

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
    //const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");
    //const cv::Mat img = cv::imread("/home/dp/Desktop/YOLO/darknet/data/dog.jpg");
    const cv::Mat img = cv::imread("/home/dp/Desktop/ARL/darknet-master/networkInput.jpg");

    graphSegmentation(img);
    //selectiveSegmentation(img);
    //contourSegmentation(img);

    cv::waitKey();

    return 0;
}