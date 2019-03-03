#include "functions.h"
#include "segmentation.h"

#include <chrono>
struct Detection {

    struct DetectionEdgeVertical {

        DetectionEdgeVertical(const cv::Point &top, int length) : top(top), height(length) {
        }

        void computeSaliencyChanges(const cv::Mat &saliency_map, int max_distance, int increment = 3) {
            CV_Assert(top.x - max_distance >= 0 && top.x + max_distance < saliency_map.cols);

            double saliency_sum_left = 0, saliency_sum_right = 0;
            int pt_cnt_left = 0, pt_cnt_right = 0;

            cv::Size increment_size(increment, height);
            for (int i = increment; i < max_distance; i += increment) {
                cv::Point left_tl = cv::Point(top.x - i, top.y);
                cv::Point right_tl = cv::Point(top.x + (i - increment), top.y);

                cv::Rect rect_left(left_tl, increment_size);
                cv::Rect rect_right(right_tl, increment_size);

                saliency_sum_left += cv::sum(saliency_map(rect_left))[0];
                saliency_sum_right += cv::sum(saliency_map(rect_right))[0];
                pt_cnt_left += rect_left.area();
                pt_cnt_right += rect_right.area();

                avg_saliency_left.emplace_back(i, (saliency_sum_left / pt_cnt_left));
                avg_saliency_right.emplace_back(i, (saliency_sum_right / pt_cnt_right));
            }

            auto comparator = [](const std::pair<int, float> &p1, const std::pair<int, float> &p2)->bool {
                return p2.second < p1.second;
            };
            std::sort(avg_saliency_left.begin(), avg_saliency_left.end(), comparator);
            std::sort(avg_saliency_right.begin(), avg_saliency_right.end(), comparator);
        }

        void draw(cv::Mat &disp_img) const {
            cv::line(disp_img, top, cv::Point(top.x, top.y + height), cv::Scalar(255, 0, 0));
        }

        cv::Point top;
        int height;
        std::vector< std::pair<int, float> > avg_saliency_left; //distance, average
        std::vector< std::pair<int, float> > avg_saliency_right; //distance, average
    };

    Detection(const cv::Mat &saliency_map, const cv::Rect &region) : box(region), avg_salicency(cv::sum(saliency_map(region))[0] / region.area()),
    left_edge(region.tl(), region.height), right_edge(cv::Point(region.tl().x + region.width, region.tl().y), region.height) {

        left_edge.computeSaliencyChanges(saliency_map, box.width * .25);
        right_edge.computeSaliencyChanges(saliency_map, box.width * .25);
    }

    cv::Rect box;
    float avg_salicency;
    DetectionEdgeVertical left_edge;
    DetectionEdgeVertical right_edge;

    static void sortBySaliency(std::vector<Detection> &detections) {
        std::sort(detections.begin(), detections.end(), [](const Detection &d1, const Detection & d2)->bool {
            return d2.avg_salicency < d1.avg_salicency;
        });
    }
};

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

    Segmentation::showSegmentationResults(img, segmentation_regions, segmentation_scores);
    cv::Mat img_saliency;

    t1 = std::chrono::high_resolution_clock::now();
    cv::saliency::StaticSaliencySpectralResidual::create()->computeSaliency(img, img_saliency);
    t2 = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << std::endl << "Saliency Map Creation Time: " << duration / 1000 << "s" << std::endl;

    cv::Mat disp_saliency;
    cv::normalize(img_saliency, disp_saliency, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    for (const cv::Rect &r : segmentation_regions)
        DrawBoundingBox(disp_saliency, r, cv::Scalar(255, 0, 0));
    DisplayImg(disp_saliency, "saliency_map");

    std::vector<Detection> detections;
    for (const cv::Rect &r : segmentation_regions)
        detections.emplace_back(img_saliency, r);

    Detection::sortBySaliency(detections);

    cv::Mat disp_img;
    cv::normalize(img_saliency, disp_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    detections[0].right_edge.draw(disp_img);
    DisplayImg("tejakljt", disp_img);

    std::cout << detections[0].avg_salicency << "|L:" << detections[0].right_edge.avg_saliency_left.front().second << "@" << detections[0].right_edge.avg_saliency_left.front().first << "|R:" << detections[0].right_edge.avg_saliency_right.front().second << "@" << detections[0].right_edge.avg_saliency_right.front().first << std::endl;

    std::vector<cv::Rect> temp_regions;
    std::vector<float> temp_scores;
    for (const Detection &d : detections) {
        temp_regions.push_back(d.box);
        temp_scores.push_back(d.avg_salicency);
    }
    Segmentation::showSegmentationResults(img, temp_regions, temp_scores, "SALIENCY_AMT");
}

int main(int argc, char * argv[]) {
    //const cv::Mat img = cv::imread("/home/dp/Downloads/ARL_data/19.png");
    //const cv::Mat img = cv::imread("/home/dp/Downloads/ARL_data/17.png");
    //const cv::Mat img = cv::imread("/home/dp/Downloads/IMG_0834.jpeg");
    cv::Mat img;
    std::srand(std::time(0));

    const char file_name_format[] = "/home/dp/Downloads/ARL_data/%d.png";
    char file_name[100];
    int file_count = 19;
    const int file_count_max = 30;

    int key = 0;
    do {
        if (key == 'n')
            ++file_count %= (file_count_max + 1);
        else if (key == 'b' && --file_count < 0)
            file_count = file_count_max;

        sprintf(file_name, file_name_format, file_count);
        img = cv::imread(file_name);

        run(img);

        key = cv::waitKey();
    } while (key != 'q' && key != 'c' && key != 27);
    cv::destroyAllWindows();

    return 0;
}