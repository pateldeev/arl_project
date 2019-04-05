#include "functions.h"
#include "yoloInterface.h"
#include "segmentation.h"
#include "saliency.h"

int main(int argc, char * argv[]) {
    const std::string labelFile = "/home/dp/Desktop/darknet-master/data/coco.names";
    const std::string configFile = "/home/dp/Desktop/darknet-master/cfg/yolov3.cfg";
    const std::string weightsFile = "/home/dp/Desktop/darknet-master/weights/yolov3.weights";
    YoloInterface yolo(configFile, weightsFile, labelFile);

    const char file_name_format[] = "/home/dp/Downloads/data_mine/%d.png";
    char file_name[100];
    int file_count = 0;
    const int file_count_max = 20;

    std::vector < cv::Rect > segmentation_regions;
    std::vector<float> segmentation_scores;
    std::time_t segmentation_srand_time = std::time(0);

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
        yolo.setThresholds(0.2);

        DisplayImg(YoloInterface::getPredictionsDisplayable(img, yolo.processImage(img)), "yolo_predictions");

        std::srand(segmentation_srand_time);
        Segmentation::process(img, segmentation_regions, segmentation_scores);
        Segmentation::showSegmentationResults(img, segmentation_regions, segmentation_scores, "my_segmentations", 2);

        cv::Mat img_saliency;
        cv::saliency::StaticSaliencySpectralResidual::create()->computeSaliency(img, img_saliency);
        cv::Mat disp_saliency;
        cv::normalize(img_saliency, disp_saliency, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::cvtColor(disp_saliency, disp_saliency, cv::COLOR_GRAY2BGR);
        std::srand(segmentation_srand_time);
        Segmentation::showSegmentationResults(disp_saliency, segmentation_regions, segmentation_scores, "saliency", 2);

        key = cv::waitKey();
    } while (key != 'q' && key != 'c' && key != 27);

    cv::destroyAllWindows();
    return 0;
}