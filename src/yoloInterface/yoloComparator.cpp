#include "functions.h"
#include "yoloInterface.h"
#include "segmentation.h"
#include "saliency.h"

int main(int argc, char * argv[]) {
    const std::string labelFile = "/home/dp/Desktop/darknet-master/data/coco.names";
    const std::string configFile = "/home/dp/Desktop/darknet-master/cfg/yolov3.cfg";
    const std::string weightsFile = "/home/dp/Desktop/darknet-master/weights/yolov3.weights";
    YoloInterface yolo(configFile, weightsFile, labelFile);

#if 0
    const char file_name_format[] = "/home/dp/Downloads/data_mine/%d.png";
    char file_name[100];
    int file_count = 3;
    const int file_count_max = 20;
#else 
    const char file_name_format[] = "/home/dp/Downloads/data_walkthrough/%d.png";
    char file_name[100];
    int file_count = 35;
    const int file_count_max = 27;
#endif
    std::vector < cv::Rect > regions;
    std::vector<float> scores;

    cv::Mat img;

    int key = 0;
    do {
        if (key == 'n')
            ++file_count %= (file_count_max + 1);
        else if (key == 'b' && --file_count < 0)
            file_count = file_count_max;

        sprintf(file_name, file_name_format, file_count);
        img = cv::imread(file_name);
        CV_Assert(!img.empty());
        yolo.setThresholds(0.25);

        Segmentation::process(img, regions, scores);

        SaliencyFilter::removeUnsalient(img, regions, scores, regions, false);

        DisplayImg(YoloInterface::getPredictionsDisplayable(img, yolo.processImage(img)), "yolo_predictions");
        Segmentation::showSegmentationResults(img, regions, scores, "my_proposals", 2);

        key = cv::waitKey();
    } while (key != 'q' && key != 'c' && key != 27);

    cv::destroyAllWindows();
    return 0;
}