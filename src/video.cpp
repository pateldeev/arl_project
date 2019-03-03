#include "functions.h"
#include "segmentation.h"

#include <opencv2/core.hpp>

int main(int argc, char * argv[]) {
    const std::string video_file_name = "/home/dp/Downloads/ARL_data_3/1.avi";
    cv::VideoCapture video(video_file_name);
    CV_Assert(video.isOpened());

    const char save_file_format[] = "/home/dp/Downloads/ARL_data_3/%d.png";
    char save_name[100];
    int save_count = 0;
    const std::vector<int> save_params = {cv::IMWRITE_PNG_COMPRESSION, 0};

    //for performing segmentation
    cv::Mat img;
    std::vector<cv::Rect> segmentation_regions;
    std::vector<float> segmentation_scores;

    //for control of operations
    unsigned int key = '\0';
    bool pause = false;

    do {
        if (key == 's') //stop and run segmentation
            pause = true;
        else if (key == 'c') //resume
            pause = false;
        else if (key == 'n')
            video >> img;
        else if (key == 'w') {
            sprintf(save_name, save_file_format, save_count++);
            cv::imwrite(save_name, img, save_params);
            std::cout << "Saved captured frame to: " << save_name << std::endl;
        }

        if (!pause) { //capture frame if requested
            video >> img;
            if (img.empty())
                break;
        } else {
            Segmentation::process(img, segmentation_regions, segmentation_scores);
            Segmentation::showSegmentationResults(img, segmentation_regions, segmentation_scores);
        }

        DisplayImg(img, "video");

        key = (pause) ? cv::waitKey() : cv::waitKey(1);
    } while (key != 'q' && key != 27); //'q' or 'ESC' to exit

    video.release();
    return 0;
}