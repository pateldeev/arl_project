
#include "yoloInterface.h"

#include "functions.h"

cv::Mat getPredictionImg(YoloInterface &y, const cv::Mat &img) {
    cv::Mat disp_img = img.clone();

    std::vector< std::pair<cv::Rect, std::pair<float, std::string>> > regions = y.processImage(img);

    int text_y_loc = 1;
    for (std::pair<cv::Rect, std::pair<float, std::string>> &r : regions) {
        cv::Scalar color(100 + std::rand() % 155, 100 + std::rand() % 155, 100 + std::rand() % 155);
        std::string text = r.second.second + "(" + std::to_string(r.second.first) + ")";
        cv::rectangle(disp_img, r.first, color, 2);
        cv::putText(disp_img, text, cv::Point(10, 25 * (++text_y_loc)), cv::FONT_HERSHEY_DUPLEX, 0.75, color, 1.25);
    }

    return disp_img;
}

int main(int argc, char * argv[]) {

    const std::string videoFile = "/home/dp/Downloads/20190125_181346.mp4";

    const std::string labelFile = "/home/dp/Desktop/darknet-master/data/coco.names";
    const std::string configFile = "/home/dp/Desktop/darknet-master/cfg/yolov3.cfg";
    const std::string weightsFile = "/home/dp/Desktop/darknet-master/weights/yolov3.weights";

    YoloInterface yolo(configFile, weightsFile, labelFile);

    //const cv::Mat img = cv::imread("/home/dp/Desktop/trainSet/Stimuli/Indoor/001.jpg");
    const cv::Mat img = cv::imread("/home/dp/Downloads/20181026_153406_HDR.jpg");
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
            DisplayImg(getPredictionImg(yolo, img_sub), "sub_image");
        } catch (...) {

        }
        key = cv::waitKey();

        std::cout << std::endl << int(key) << std::endl;
    } while (char(key) != 'c' || char(key) != 'q');



    cv::waitKey();
    return 0;

    cv::VideoCapture data(videoFile);
    CV_Assert(data.isOpened());

    cv::Mat frame;
    data >> frame;
    DisplayImg(frame, "test");

    while (data.read(frame)) {
        cv::waitKey(1);
        UpdateImg(getPredictionImg(yolo, frame), "test");
    }

    return 0;
}