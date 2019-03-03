#include "functions.h"
#include "segmentation.h"

#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#define DEBUG
#ifdef DEBUG
static bool disp_details = false;
#endif

void calculateMeanSTD(std::vector<float> &data, float &mean, float &stdeviation) {
    float sum = 0.0;
    stdeviation = 0.0;
    for (float d : data)
        sum += d;
    mean = sum / data.size();

    for (float d : data)
        stdeviation += std::pow(d - mean, 2);

    stdeviation = std::sqrt(stdeviation / data.size());
}

struct Detection {

    Detection(const cv::Mat &saliency_map, const cv::Rect &region, float std_threshold = 1.25) : m_box(region), m_img_size(saliency_map.cols, saliency_map.rows), m_avg_saliency(cv::mean(saliency_map(region))[0]), m_status(1), m_std_thresh(std_threshold) {
    }

    void ensureRegionSaliency(const cv::Mat &saliency_map, float saliency_map_mean, float saliency_map_std) {
        //ensure region is salient enough
        if (m_status == 1 && (((m_avg_saliency - saliency_map_mean) / saliency_map_std) < m_std_thresh))
            m_status = -1; //remove any region that is not salient compared to entire image

        //ensure region stands out from immediate surroundings (x2 size)- will bring back if region stands out a lot
        cv::Point doubled_tl(std::max(0, m_box.tl().x - (m_box.width / 2)), std::max(0, m_box.tl().y - (m_box.height / 2)));
        cv::Point doubled_br(std::min(saliency_map.cols - 1, m_box.br().x + (m_box.width / 2)), std::min(saliency_map.rows - 1, m_box.br().y + (m_box.height / 2)));
        cv::Rect region_doubled(doubled_tl, doubled_br);
        float doubled_saliency = cv::sum(saliency_map(region_doubled))[0];
        float surrounding_avg_saliency = (doubled_saliency - (m_avg_saliency * m_box.area())) / (region_doubled.area() - m_box.area());

#ifdef DEBUG
        if (disp_details) {
            cv::Mat disp = saliency_map.clone();
            DrawBoundingBox(disp, m_box, cv::Scalar(255, 0, 0));
            DrawBoundingBox(disp, region_doubled, cv::Scalar(255, 0, 0));
            std::cout << "Region|Surrounding:" << m_avg_saliency << "|" << surrounding_avg_saliency << std::endl;
            DisplayImg(disp, "DEBUG");
            cv::waitKey();
        }
#endif

        if (m_status == 1 && surrounding_avg_saliency > 0.8 * m_avg_saliency)
            m_status = -2; //remove regions that are not salient compared to surroundings
        else if (m_status == -1 && m_avg_saliency > 2.5 * surrounding_avg_saliency)
            m_status = 1; //restore regions that are very salient compared to surroundings
    }

    void expandTop(const cv::Mat &saliency_map, const cv::Mat &img) {
        const float max_expansion_percentage = 0.25;
        const int expansion_interval = 3;

        cv::Point left = m_box.tl(), right = left + cv::Point(m_box.width, 0);

        const int max_expansion_down = max_expansion_percentage * m_box.height;
        const int max_expansion_up = std::min(max_expansion_down, left.y);

        cv::Mat disp = img.clone();
        cv::line(disp, left, right, cv::Scalar(255, 0, 0));
        cv::line(disp, left - cv::Point(0, max_expansion_up), right - cv::Point(0, max_expansion_up), cv::Scalar(0, 0, 255));
        cv::line(disp, left + cv::Point(0, max_expansion_down), right + cv::Point(0, max_expansion_down), cv::Scalar(0, 255, 0));

        //try to expand upwards
        for (int i = expansion_interval; i <= max_expansion_up; i += expansion_interval) {
            float avg_saliency = cv::mean(saliency_map(cv::Range(left.y - i, left.y), cv::Range(left.x, right.x)))[0];
            std::cout << std::endl << m_avg_saliency << '|' << avg_saliency << std::endl;
            break;
        }

        //try to expand downwards
        for (int i = expansion_interval; i <= max_expansion_down; i += expansion_interval) {
            float avg_saliency = cv::mean(saliency_map(cv::Range(left.y, left.y + i), cv::Range(left.x, right.x)))[0];
            std::cout << std::endl << m_avg_saliency << '|' << avg_saliency << std::endl;

            break;
        }

        DisplayImg(disp, "Expand_Top");
    }

    void peformMerging(Detection &smaller, const cv::Rect &overlap, float saliency_map_std, float saliency_std_thresh = 1.f) {
        float surrounding_avg_saliency = (m_avg_saliency * m_box.area() - smaller.m_avg_saliency * overlap.area()) / (m_box.area() - overlap.area());
        float difference_std = (smaller.m_avg_saliency - surrounding_avg_saliency) / saliency_map_std;

        if (difference_std > saliency_std_thresh) {
            m_status = 0; //merge into smaller region - it is much more salient
        } else {
            smaller.m_status = 0; //merge into bigger region - smaller region is not special
        }
    }

    cv::Rect m_box;
    cv::Size m_img_size;
    float m_avg_saliency;
    int m_status; //1 = fine, 0 = merged with another detection, -1 = removed by standard deviation too low, -2 = removed because too much like neighbors
    float m_std_thresh; //used to determine standard deviation threshold and includes merges based on similar saliency

    //sorts from high to low

    static void sortByArea(std::vector<Detection> &detections) {
        std::sort(detections.begin(), detections.end(), [](const Detection &d1, const Detection & d2)->bool {
            return d2.m_box.area() < d1.m_box.area();
        });
    }

    static void mergeBySaliency(std::vector<Detection> &detections, float saliency_map_std, float coverage_thresh = 0.88, float saliency_std_thresh = 1.f) {
        sortByArea(detections); //sort detections by area - high to low

        //go through and try to merge each region with any that are smaller than it
        for (int i = 0; i < detections.size(); ++i) {
            if (detections[i].m_status != 1)
                continue; //only consider valid regions

            for (int j = i + 1; j < detections.size(); ++j) {
                if (detections[j].m_status != 1)
                    continue; //only consider valid regions

                //if there is enough area overlap, merge regions
                cv::Rect overlap = detections[i].m_box & detections[j].m_box;
                if (overlap.area() > 0.5 * detections[i].m_box.area() && overlap.area() > coverage_thresh * detections[j].m_box.area()) {
                    detections[i].peformMerging(detections[j], overlap, saliency_map_std, saliency_std_thresh);

                    if (detections[i].m_status == 0)
                        break; //region merged - no longer valid
                }
            }
        }
    }
};

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

    cv::Mat img_saliency;
    t1 = std::chrono::high_resolution_clock::now();
    cv::saliency::StaticSaliencySpectralResidual::create()->computeSaliency(img, img_saliency);
    t2 = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << std::endl << "Saliency Map Creation Time: " << duration / 1000 << "s" << std::endl;

    Segmentation::showSegmentationResults(img, segmentation_regions, segmentation_scores, "regions_seg_scores");

    for (const cv::Rect &r : segmentation_regions)
        avg_saliency.push_back(cv::mean(img_saliency(r))[0]);

    Segmentation::showSegmentationResults(img, segmentation_regions, avg_saliency, "regions_avg_saliency");

    float segmentation_scores_mean, segmentation_scores_std;
    calculateMeanSTD(segmentation_scores, segmentation_scores_mean, segmentation_scores_std);
    std::cout << "Segmentation Scores mean|std_dev: " << segmentation_scores_mean << "|" << segmentation_scores_std << std::endl;

    cv::Mat disp_saliency;
    cv::normalize(img_saliency, disp_saliency, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    for (const cv::Rect &r : segmentation_regions)
        DrawBoundingBox(disp_saliency, r, cv::Scalar(255, 0, 0));

    DisplayImg(disp_saliency, "saliency_map");

    std::vector<Detection> detections;
    cv::Scalar saliency_mean, saliency_std;
    cv::meanStdDev(img_saliency, saliency_mean, saliency_std);
    std::cout << std::endl << "Img saliency mean|std_dev: " << saliency_mean[0] << "|" << saliency_std[0] << std::endl;

    float base_saliency_std_threshold = 1.25;
    for (int i = 0; i < segmentation_regions.size(); ++i) {
        float std_threshold = base_saliency_std_threshold;

        float segmention_score_stddeviation = (segmentation_scores[i] - segmentation_scores_mean) / segmentation_scores_std;
        if (segmention_score_stddeviation > 1.0)
            std_threshold -= segmention_score_stddeviation / 2;

        detections.emplace_back(img_saliency, segmentation_regions[i], std_threshold);
    }

    Detection::mergeBySaliency(detections, saliency_std[0]);

    std::vector<cv::Rect> surviving;
    for (Detection &d : detections) {
        if (d.m_status == 1)
            surviving.emplace_back(d.m_box);
    }
    cv::Mat disp = img.clone();
    DrawBoundingBoxes(disp, surviving);
    DisplayImg(disp, "Non-merged");

    for (Detection &d : detections) {
        d.ensureRegionSaliency(img_saliency, saliency_mean[0], saliency_std[0]);
        //d.expandTop(img_saliency, img);
        //cv::waitKey();
    }

    surviving.clear();
    for (Detection &d : detections) {
        if (d.m_status == 1)
            surviving.emplace_back(d.m_box);
    }
    disp = img.clone();
    DrawBoundingBoxes(disp, surviving);
    DisplayImg(disp, "SURVIVING");
}

int main(int argc, char * argv[]) {
    //const cv::Mat img = cv::imread("/home/dp/Downloads/ARL_data/19.png");
    //const cv::Mat img = cv::imread("/home/dp/Downloads/ARL_data/17.png");
    //const cv::Mat img = cv::imread("/home/dp/Downloads/IMG_0834.jpeg");
    cv::Mat img;
    std::srand(std::time(0));

    const char file_name_format[] = "/home/dp/Downloads/data_cumulative/%d.png";
    char file_name[100];
    int file_count = 0;
    const int file_count_max = 18;

    int key = 0;
    do {
        if (key == 'n')
            ++file_count %= (file_count_max + 1);
        else if (key == 'b' && --file_count < 0)
            file_count = file_count_max;

        sprintf(file_name, file_name_format, file_count);
        img = cv::imread(file_name);
        img = cv::imread("/home/dp/Downloads/P1020702.JPG");

        run(img);

        key = cv::waitKey();

#ifdef DEBUG
        if (key == 'd')
            disp_details = true;
        else if (key == 'r')
            disp_details = false;
#endif

    } while (key != 'q' && key != 'c' && key != 27);
    cv::destroyAllWindows();

    return 0;
}