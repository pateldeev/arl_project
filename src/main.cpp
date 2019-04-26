#include "functions.h"
#include "segmentation.h"
#include "saliency.h"
#include "yoloInterface.h"

int main(int argc, char * argv[]) {
    const cv::Mat img = cv::imread("/home/dp/Downloads/poster/Img01/img.png");
    CV_Assert(!img.empty());

    std::vector<cv::Rect> segmentation_regions;
    std::vector<float> segmentation_scores;
    Segmentation::process(img, segmentation_regions, segmentation_scores);

    std::vector<cv::Rect> surviving_regions;
    SaliencyFilter::removeUnsalient(img, segmentation_regions, segmentation_scores, surviving_regions, false);

    cv::Mat r0 = img(surviving_regions[0]).clone(), r1 = img(surviving_regions[1]).clone(), r2 = img(surviving_regions[2]).clone();

    cv::Mat disp = img.clone();
    for (const cv::Rect &r : surviving_regions)
        DrawBoundingBox(disp, r, cv::Scalar(0, 0, 255), false, 3);

    YoloInterface y(0.1);
    //y.loadNetwork("/home/dp/Desktop/darknet-master/cfg/yolov3.cfg", "/home/dp/Desktop/darknet-master/weights/yolov3.weights", "/home/dp/Desktop/darknet-master/data/coco.names");
    //y.processImage(img);
    //y.saveResults("/home/dp/Downloads/poster/Img01/yolo_results.txt");

    y.readResults("/home/dp/Downloads/poster/Img01/yolo_results.txt");
    DrawBoundingBox(disp, y[1].first, cv::Scalar(0, 255, 0), false, 3);
    DrawBoundingBox(disp, y[2].first, cv::Scalar(0, 255, 0), false, 3);
    DrawBoundingBox(disp, y[3].first, cv::Scalar(0, 255, 0), false, 3);


    cv::Mat disp_my = cv::Mat(img.rows, img.cols, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::resize(r2, r2, cv::Size(3.5 * r2.cols, 3.5 * r2.rows));
    r2.copyTo(disp_my(cv::Rect(8, 75, r2.cols, r2.rows)));

    cv::resize(r0, r0, cv::Size(3.5 * r0.cols, 3.5 * r0.rows));
    r0.copyTo(disp_my(cv::Rect(8, 265, r0.cols, r0.rows)));

    cv::resize(r1, r1, cv::Size(3.5 * r1.cols, 3.5 * r1.rows));
    r1.copyTo(disp_my(cv::Rect(8, 900, r1.cols, r1.rows)));



    cv::Mat disp_yolo = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);

    std::vector<cv::Mat> test = {disp_yolo, disp, disp_my};
    DisplayMultipleImages("Results", test, 1, 3);

    //cv::Mat final_results = Segmentation::showSegmentationResults(img, surviving_regions, segmentation_scores, "FINAL_PROPOSALS", 2, false);
    cv::waitKey();
    return 0;
}