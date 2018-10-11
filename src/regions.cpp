#include "functions.h"

#include <iostream>

void displayRegions(const cv::Mat & img, const std::vector<cv::Rect> regions, const std::string & windowName = "Regions_Img", const int increment = 50) {
    cv::namedWindow(windowName.c_str(), cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName.c_str(), 1200, 1200);

    int numRectsShown = increment; // number of region proposals to show

    while (1) {
        cv::Mat dispImg = img.clone();

        // iterate over all the region proposals
        for (int i = 0; i < regions.size(); ++i) {
            if (i < numRectsShown) {
                cv::rectangle(dispImg, regions[i], cv::Scalar(0, 255, 0));
            } else {
                break;
            }
        }

        cv::imshow(windowName.c_str(), dispImg);

        int k = cv::waitKey(); // record key press

        if (k == 'm')
            numRectsShown += increment; // increase total number of rectangles to show by increment
        else if (k == 'l' && numRectsShown > increment)
            numRectsShown -= increment; // decrease total number of rectangles to show by increment
        else if (k == 'q')
            return;
    }
}

int main(int argc, char * argv[]) {

    cv::namedWindow("Orignal_Img", cv::WINDOW_NORMAL);
    cv::resizeWindow("Orignal_Img", 1200, 1200);

    // read image
    const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");
    //const cv::Mat img = cv::imread("/home/dp/Downloads/NFPA_dataset/NFPA dataset/pos-175.jpg");

    std::vector<cv::Rect> regions;

    CalculateRegionProposals(img, regions);

    std::cout << std::endl << "Total Number of Region Proposals: " << regions.size() << std::endl;

    cv::imshow("Orignal_Img", img);
    displayRegions(img, regions);

    return 0;
}
