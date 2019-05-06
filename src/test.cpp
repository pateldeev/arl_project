#include "functions.h"
#include "segmentation.h"
#include "saliency.h"

#include <chrono>

void run(const cv::Mat &img) {
    std::vector<cv::Rect> segmentation_regions;
    std::vector<float> segmentation_scores;

    std::chrono::high_resolution_clock::time_point t1, t2;
    double duration;

    t1 = std::chrono::high_resolution_clock::now();
    Segmentation::process(img, segmentation_regions, segmentation_scores);
    t2 = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << std::endl << "Segmentation Time: " << duration / 1000 << "s" << std::endl;

    std::vector<cv::Rect> surviving_regions;
    SaliencyFilter::removeUnsalient(img, segmentation_regions, segmentation_scores, surviving_regions, true);

    cv::Mat final_results = Segmentation::showSegmentationResults(img, surviving_regions, segmentation_scores, "FINAL_PROPOSALS", 3, false);
    //SaveImg(final_results, "/home/dp/Downloads/poster/saliency/original_segs");
}

int main(int argc, char * argv[]) {
    cv::Mat img;
    char file_name_format[100];
    char file_name[100];
    int file_max;
    int file_current = 0;

    char data_set = '0';
    if (argc > 1)
        data_set = argv[1][0];

    if (data_set == '1') {
        strcpy(file_name_format, "/home/dp/Downloads/data_walkthrough/%d.png");
        file_current = 27;
        file_max = 36;
    } else if (data_set == '2') {
        strcpy(file_name_format, "/home/dp/Downloads/data_mine/%d.png");
        file_current = 4;
        file_max = 20;
    } else if (data_set == '3') {
        strcpy(file_name_format, "/home/dp/Downloads/data_cumulative/%d.png");
        file_max = 17;
    } else if (data_set == '4') {
        strcpy(file_name_format, "/home/dp/Downloads/test/%d.jpg");
        file_max = 10;
    } else if (data_set == '5') {
        strcpy(file_name_format, "/home/dp/Downloads/data_new/%d.png");
        file_max = 10;
    } else {
        strcpy(file_name_format, "/home/dp/Downloads/data_representative/%d.png");
        file_max = 14;
        //file_current = 12;
    }
    if (argc > 2)
        file_current = std::stoi(argv[2]);

    int key = 0;
    do {
        if (key == 'n')
            ++file_current %= (file_max);
        else if (key == 'b' && --file_current < 0)
            file_current = file_max;

        sprintf(file_name, file_name_format, file_current);
        img = cv::imread(file_name);
        img = cv::imread("/home/dp/Downloads/poster/Img07/img.png");
        //img = cv::imread("/home/dp/Downloads/data_mine/8.png");
        CV_Assert(!img.empty());

        run(img);

        key = cv::waitKey();
        cv::destroyAllWindows();

    } while (key != 'q' && key != 'c' && key != 27);
    cv::destroyAllWindows();

    return 0;
}