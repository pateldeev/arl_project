#include "functions.h"
#include "segmentation.h"
#include "saliency.h"

#include <chrono>

void run(const cv::Mat &img) {
    std::vector<cv::Rect> segmentation_regions;
    std::vector<float> segmentation_scores, avg_saliency;

    std::chrono::high_resolution_clock::time_point t1, t2;
    double duration;

    t1 = std::chrono::high_resolution_clock::now();
    Segmentation::process(img, segmentation_regions, segmentation_scores);
    t2 = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << std::endl << "Segmentation Time: " << duration / 1000 << "s" << std::endl;

    std::vector<cv::Rect> surviving_regions;
    SaliencyFilter::removeUnsalient(img, segmentation_regions, segmentation_scores, surviving_regions);
}

int main(int argc, char * argv[]) {
    //const cv::Mat img = cv::imread("/home/dp/Downloads/ARL_data/19.png");
    //const cv::Mat img = cv::imread("/home/dp/Downloads/ARL_data/17.png");
    //const cv::Mat img = cv::imread("/home/dp/Downloads/IMG_0834.jpeg");
    cv::Mat img;
    std::srand(std::time(0));
#if 0
    const char file_name_format[] = "/home/dp/Downloads/data_cumulative/%d.png";
    char file_name[100];
    int file_count = 0;
    const int file_count_max = 18;
#else
    const char file_name_format[] = "/home/dp/Downloads/data_mine/%d.png";
    char file_name[100];
    int file_count = 0;
    const int file_count_max = 20;
#endif

    int key = 0;
    do {
        if (key == 'n')
            ++file_count %= (file_count_max + 1);
        else if (key == 'b' && --file_count < 0)
            file_count = file_count_max;

        sprintf(file_name, file_name_format, file_count);
        img = cv::imread(file_name);
        //img = cv::imread("/home/dp/Downloads/images_Gonzen_shorten_FLIGHT_LONG_HOMINGEND/frame0636.jpg");

        run(img);

        key = cv::waitKey();
        cv::destroyAllWindows();

    } while (key != 'q' && key != 'c' && key != 27);
    cv::destroyAllWindows();

    return 0;
}