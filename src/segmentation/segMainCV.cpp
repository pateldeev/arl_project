#include "functions.h"

#include "segmentationCV.h"

#include <chrono>

int main(int argc, char * argv[]) {
    //const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");
    //const cv::Mat img = cv::imread("/home/dp/Desktop/ARL/darknet-master/networkInput.jpg");
    const cv::Mat img = cv::imread("/home/dp/Downloads/20181108_190017_HDR.jpg");
    //const cv::Mat img = cv::imread("/home/dp/Downloads/IMG_0834.jpeg");
    //const cv::Mat img = cv::imread("/home/dp/Desktop/Screenshot 2019-02-07 19:15:42.png");
#if 1
    std::vector<cv::Rect> proposals;

    //DisplayImg(img, "orignal");

    //rescale image
    const int resizeHeight = 200;
    const int oldHeight = img.rows, oldWidth = img.cols;
    const int resizeWidth = oldWidth * resizeHeight / oldHeight;
    cv::Mat imgResized;
    cv::resize(img, imgResized, cv::Size(resizeWidth, resizeHeight));
    DisplayImg(imgResized, "resized");

    std::chrono::high_resolution_clock::time_point t1, t2;
    t1 = std::chrono::high_resolution_clock::now();

    SegmentationCV::process(imgResized, proposals, 150, 150, 0.8);

    t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    //rescale rectangles back for original image
    for (cv::Rect & rec : proposals) {
        int x = oldWidth * (double(rec.x) / resizeWidth);
        int y = oldHeight * (double(rec.y) / resizeHeight);
        int width = oldWidth * (double(rec.width) / resizeWidth);
        int height = oldHeight * (double(rec.height) / resizeHeight);

        rec = cv::Rect(x, y, width, height);
    }

    std::cout << std::endl << "Region Proposals: " << proposals.size() << std::endl << "Time: " << duration << std::endl;

    std::sort(proposals.begin(), proposals.end(), [](const cv::Rect & r1, const cv::Rect & r2)->bool {
        return r1.area() < r2.area();
    });

    DisplayBoundingBoxesInteractive(img, proposals, "All Proposals");

    std::vector<cv::Rect> proposalsFiltered(proposals);
    RemoveOverlapping(proposalsFiltered);
    std::cout << std::endl << "Filtered Region Proposals: " << proposalsFiltered.size() << std::endl;

    //DisplayBoundingBoxesInteractive(img, proposalsFiltered, "Filtered Proposals");

#ifdef DEBUG_SEGMENTATION    
    DisplayMultipleImages("m_images", SegmentationCV::m_images, 2, 3);
    std::vector<cv::Mat> temp1, temp2;
    for (unsigned int i = 0; i < SegmentationCV::debugDisp1.size(); ++i) {
        temp1.push_back(GetGraphSegmentationViewable(SegmentationCV::debugDisp1[i]));
        temp2.push_back(SegmentationCV::debugDisp2[i]);
    }
    unsigned int key = '-';
    int currentView = 1;
    do {
        if (key == '+' || key == '=') {
            ++currentView;
            currentView %= 5;
        } else if (key == '-') {
            if (currentView > 0)
                --currentView;
            else
                currentView = 4;
        }

        std::vector<cv::Mat> tempDisp1, tempDisp2;
        tempDisp1.push_back(SegmentationCV::m_images[currentView].clone());
        tempDisp2.push_back(SegmentationCV::m_images[currentView].clone());
        for (unsigned int i = currentView * 5; i < currentView * 5 + 5; ++i) {
            tempDisp1.push_back(temp1[i].clone());
            tempDisp2.push_back(temp2[i].clone());
        }

        DisplayMultipleImages("m_images[]_regions", tempDisp2, 2, 3);
        DisplayMultipleImages("m_images[]", tempDisp1, 2, 3);
        key = cv::waitKey();
    } while (key != 'c' && key != 'q');
    cv::destroyAllWindows();
#endif

    return 0;

    cv::Mat heatMap(img.rows, img.cols, CV_16UC1, cv::Scalar(0));
    CreateHeatMap(proposals, heatMap);
    cv::Mat tempDisp;
    cv::normalize(heatMap, tempDisp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    DisplayImg(tempDisp, "Heatmap");
    cv::waitKey();
#else    
    std::vector<cv::Mat> test;
    for (int i = 0; i < 10; ++i)
        test.push_back(img.clone());

    //DisplayImg(test[0], "test", test[0].rows, test[0].cols, true);

    DisplayMultipleImages("test", test);

    cv::waitKey();
#endif

    return 0;
}