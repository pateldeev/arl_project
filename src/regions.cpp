#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/ximgproc/segmentation.hpp>

#include <iostream>

int main(int argc, char * argv[]) {

    cv::namedWindow("Orignal_Img", cv::WINDOW_NORMAL);
    cv::resizeWindow("Orignal_Img", 1200, 1200);
    cv::namedWindow("Regions_origImg", cv::WINDOW_NORMAL);
    cv::resizeWindow("Regions_origImg", 1200, 1200);
    cv::namedWindow("Regions_scaledImg", cv::WINDOW_NORMAL);
    cv::resizeWindow("Regions_scaledImg", 1200, 1200);

    // speed-up using multithreads
    cv::setUseOptimized(true);
    cv::setNumThreads(4);

    // read image
    const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");
    cv::imshow("Orignal_Img", img);

    // resize image to reduce computation
    const int newHeight = 200;
    const int oldHeight = img.rows;
    const int oldWidth = img.cols;
    const int newWidth = oldWidth * newHeight / oldHeight;
    cv::Mat imgResized;
    cv::resize(img, imgResized, cv::Size(newWidth, newHeight));

    // create Selective Search Segmentation Object using default parameters
    cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();

    ss->setBaseImage(imgResized); // set input image on which we will run segmentation

    ss->switchToSelectiveSearchQuality(); // change to high quality

    // run selective search segmentation on input image
    std::vector<cv::Rect> rectResized, rectOriginal;
    ss->process(rectResized);

    // rescale rectangles back for original image
    for (const cv::Rect & rec : rectResized) {
        int x = oldWidth * ((double) rec.x / newWidth);
        int y = oldHeight * ((double) rec.y / newHeight);
        int width = ((double) rec.width / newWidth) * oldWidth;
        int height = ((double) rec.height / newHeight) * oldHeight;

        rectOriginal.push_back(cv::Rect(x, y, width, height));
    }
    std::cout << std::endl << "Total Number of Region Proposals: " << rectResized.size() << std::endl;

    int numShowRects = 100; // number of region proposals to show
    const int increment = 50; // increment to increase/decrease total number of reason proposals to be shown

    while (1) {
        cv::Mat imgOutOriginal = img.clone();
        cv::Mat imgOutScaled = imgResized.clone(); 

        // iterate over all the region proposals
        for (int i = 0; i < rectResized.size(); ++i) {
            if (i < numShowRects) {
                cv::rectangle(imgOutOriginal, rectOriginal[i], cv::Scalar(0, 255, 0));
		cv::rectangle(imgOutScaled, rectResized[i], cv::Scalar(0,255,0));
            } else {
                break;
            }
        }

        cv::imshow("Regions_origImg", imgOutOriginal);
	cv::imshow("Regions_scaledImg", imgOutScaled);

        int k = cv::waitKey(); // record key press

        if (k == 'm') {
            numShowRects += increment; // increase total number of rectangles to show by increment
        } else if (k == 'l' && numShowRects > increment) {
            numShowRects -= increment; // decrease total number of rectangles to show by increment
        } else if (k == 'q') {
            break;
        }
    }

    return 0;
}
