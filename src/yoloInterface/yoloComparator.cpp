#include "functions.h"
#include "yoloInterface.h"
#include "segmentation.h"
#include "saliency.h"

int main(int argc, char* argv[]) {
    const std::string labelFile = "/home/dp/Desktop/darknet-master/data/coco.names";
    const std::string configFile = "/home/dp/Desktop/darknet-master/cfg/yolov3.cfg";
    const std::string weightsFile = "/home/dp/Desktop/darknet-master/weights/yolov3.weights";
    YoloInterface yolo(configFile, weightsFile, labelFile);

#if 0
    const char file_name_format[] = "/home/dp/Downloads/data_mine/%d.png";
    char file_name[100];
    int file_current = 3;
    const int file_count_max = 20;
#elif 1
    const char file_name_format[] = "/home/dp/Downloads/data_new/%d.png";
    char file_name[100];
    int file_current = 0;
    const int file_count_max = 10;
#else 
    const char file_name_format[] = "/home/dp/Downloads/data_walkthrough/%d.png";
    char file_name[100];
    int file_current = 35;
    const int file_count_max = 27;
#endif
    if (argc > 1)
        file_current = std::stoi(argv[1]);

    std::vector < cv::Rect > regions;
    std::vector<float> scores;

    cv::Mat img;

    int key = 0;
    do {
        if (key == 'n')
            ++file_current %= (file_count_max + 1);
        else if (key == 'b' && --file_current < 0)
            file_current = file_count_max;

        sprintf(file_name, file_name_format, file_current);
        img = cv::imread(file_name);
        img = cv::imread("/home/dp/Downloads/poster/Img11/img.png");
        CV_Assert(!img.empty());

        Segmentation::process(img, regions, scores);

        SaliencyFilter::removeUnsalient(img, regions, scores, regions, false);

        yolo.setThresholds(0.001);
        cv::Mat img_yolo_results = YoloInterface::getPredictionsDisplayable(img, yolo.processImage(img));
        DisplayImg(img_yolo_results, "yolo_predictions");
        SaveImg(img_yolo_results, "/home/dp/Downloads/poster/Img11/yolo.png");
        yolo.saveResults("/home/dp/Downloads/poster/Img11/yolo_results.txt");

        Segmentation::showSegmentationResults(img, regions, scores, "my_proposals", 2);

        key = cv::waitKey();
    } while (key != 'q' && key != 'c' && key != 27);

    cv::destroyAllWindows();
    return 0;
}