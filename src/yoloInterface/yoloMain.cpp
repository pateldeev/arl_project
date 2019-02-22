
#include <opencv2/highgui.hpp>

#include "yoloInterface.h"

#include "functions.h"

cv::Mat getPredictionImg(YoloInterface &y, const cv::Mat &img) {
    return YoloInterface::getPredictionsDisplayable(img, y.processImage(img));
}

int main(int argc, char * argv[]) {
    const std::string videoFile = "/home/dp/Downloads/20190125_181346.mp4";

    const std::string labelFile = "/home/dp/Desktop/darknet-master/data/coco.names";
    const std::string configFile = "/home/dp/Desktop/darknet-master/cfg/yolov3.cfg";
    const std::string weightsFile = "/home/dp/Desktop/darknet-master/weights/yolov3.weights";

    YoloInterface yolo(configFile, weightsFile, labelFile);

    //const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");
    //const cv::Mat img = cv::imread("/home/dp/Downloads/20181108_190017_HDR.jpg");
    const cv::Mat img = cv::imread("/home/dp/Downloads/IMG_0834.jpeg");
    DisplayImg(getPredictionImg(yolo, img), "full");

    int rStart = 100, cStart = 700;

    unsigned int key;
    do {
        if (key == 84)
            rStart += 10;
        else if (key == 82)
            rStart -= 10;
        else if (key == 81)
            cStart -= 10;
        else if (key == 83)
            cStart += 10;

        try {
            const cv::Mat img_sub = img(cv::Range(rStart, rStart + 608), cv::Range(cStart, cStart + 608));
            DisplayImg(getPredictionImg(yolo, img_sub), "sub_img");
        } catch (...) {
            std::cerr << "Error: out of range!" << std::endl;
        }

        key = cv::waitKey();

        std::cout << std::endl << int(key) << std::endl;
    } while (key != 'c' && key != 'q');
    cv::destroyAllWindows();
    return 0;

    cv::VideoCapture data(videoFile);
    CV_Assert(data.isOpened());

    cv::Mat frame;
    data >> frame;
    DisplayImg(frame, "test");

    while (data.read(frame)) {
        if (cv::waitKey(1) == 'q') break;

        UpdateImg(getPredictionImg(yolo, frame), "test");
    }

    return 0;
}