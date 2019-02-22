#include "functions.h"

#include "segmentation.h"

#include <iostream>
#include <chrono>
#include <opencv2/highgui.hpp>

void showSegmentations(const cv::Mat &img, const std::vector<cv::Mat> &segmentations, const std::vector< std::vector<cv::Rect> > &regions) {
    std::vector<cv::Mat> disp_segs = {img}, disp_regions = {img.clone()};

    for (const cv::Mat &s : segmentations)
        disp_segs.push_back(GetGraphSegmentationViewable(s, true));

    int region_cnt = 0;
    for (const std::vector<cv::Rect> &r : regions) {
        disp_regions.emplace_back(img.clone());
        DrawBoundingBoxes(disp_regions.back(), r);
        WriteText(disp_regions.back(), std::to_string(r.size()));
        region_cnt += r.size();
    }
    WriteText(disp_regions[0], std::to_string(region_cnt));

    ShowManyImages("segmentations", disp_segs, 2, 3);
    ShowManyImages("proposals", disp_regions, 2, 3);
}

void showSegmentationStats(const cv::Mat &img, const cv::Mat &img_seg) {
    double min, max;
    cv::minMaxLoc(img_seg, &min, &max);
    int num_segs = max + 1;

    int disp_seg = 0;
    int key = 0;

    std::vector<cv::Rect> regions;
    Segmentation::getBoundingBoxes(img_seg, regions);

    cv::Mat disp_img = img.clone();
    DrawBoundingBoxes(disp_img, regions);
    DisplayImg(disp_img, "regions");

    do {
        if ((key == '+' || key == '=') && ++disp_seg >= num_segs)
            disp_seg = 0;
        else if (key == '-' && --disp_seg < 0)
            disp_seg = num_segs - 1;

        disp_img = img.clone();

        const uint32_t * ptr_seg;
        uint8_t * ptr_disp;

        for (int i = 0; i < img_seg.rows; ++i) {
            ptr_seg = img_seg.ptr<uint32_t>(i);
            ptr_disp = disp_img.ptr<uint8_t>(i);

            for (int j = 0; j < img_seg.cols; ++j) {

                if (ptr_seg[j] == disp_seg) {
                    cv::Scalar color(0, 0, 0);
                    ptr_disp[j * 3] = uint8_t(color[0]);
                    ptr_disp[j * 3 + 1] = uint8_t(color[1]);
                    ptr_disp[j * 3 + 2] = uint8_t(color[2]);
                }
            }
        }
        cv::putText(disp_img, std::to_string(int(disp_seg)), cv::Point(10, 25), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 255), 1.25);

        key = DisplayImg(disp_img, "details", 1200, 1200, true);
    } while (key != 'q' && key != 'c' && key != 27);
    cv::destroyWindow("details");
    cv::destroyWindow("regions");
}

void showRegions(const cv::Mat &img, const std::vector< std::pair<cv::Rect, float> > &regions_sorted) {
    cv::Mat disp_all = img.clone(), disp_highest = img.clone();
    for (const std::pair<cv::Rect, float> &r : regions_sorted)
        DrawBoundingBox(disp_all, r.first, cv::Scalar(0, 255, 0), false);

    std::vector< std::pair<cv::Rect, float> >::const_iterator region_highest = regions_sorted.end();
    DrawBoundingBox(disp_highest, (--region_highest)->first, cv::Scalar(0, 0, 255), false);
    std::cout << std::endl << "Highest: " << region_highest->first << ",";
    DrawBoundingBox(disp_highest, (--region_highest)->first, cv::Scalar(255, 0, 0), false);
    std::cout << region_highest->first << ",";
    DrawBoundingBox(disp_highest, (--region_highest)->first, cv::Scalar(0, 255, 0), false);
    std::cout << region_highest->first << std::endl;

    DisplayImg("surviving_regions", disp_all);
    DisplayImg("surviving_regions_highest(R>B>G)", disp_highest);

    return;
}

int main(int argc, char * argv[]) {
#if 0
    const char file_name_format[] = "/home/dp/Downloads/ARL_data/%d.png";
    char file_name[100];
    int file_count = 19;
    const int file_count_max = 30;
#else 
    const char file_name_format[] = "/home/dp/Downloads/data_2/%d.png";
    char file_name[100];
    int file_count = 0;
    const int file_count_max = 15;
#endif

    cv::Mat img;
    const std::vector<float> k_values = {500, 600, 700, 800, 900};
    std::vector<cv::Mat> img_domains;
    std::vector< std::vector<cv::Mat> > img_segmentations;
    std::vector< std::vector< std::vector<cv::Rect> > > img_regions; //[domain, seg -- regions]
    std::vector< std::pair<cv::Rect, float> > regions_significant;
    int disp_domain = 0;

    std::chrono::high_resolution_clock::time_point t1, t2;
    double duration;

    int key = 0;
    do {
        if (key == 'n')
            ++file_count %= (file_count_max + 1);
        else if (key == 'b' && --file_count < 0)
            file_count = file_count_max;
        else if (key == '+' || key == '=' || key == 171)
            ++disp_domain %= img_domains.size();
        else if ((key == '-' || key == 173) && --disp_domain < 0)
            disp_domain += img_domains.size();
        else if (key == '1' || key == '2' || key == '3' || key == '4' || key == '5')
            showSegmentationStats(img_domains[0], img_segmentations[disp_domain][key - '1']);

        if (key == '+' || key == '=' || key == '-' || key == 171 || key == 173) {
            showSegmentations(img_domains[disp_domain], img_segmentations[disp_domain], img_regions[disp_domain]);
        } else {
            sprintf(file_name, file_name_format, file_count);
            img = cv::imread(file_name);
            //img = cv::imread("/home/dp/Downloads/IMG_0834.jpeg");

            t1 = std::chrono::high_resolution_clock::now();

            Segmentation::getDomains(img, img_domains);
            Segmentation::getSegmentations(img_domains, img_segmentations, k_values);
            img_regions.resize(img_segmentations.size(), std::vector< std::vector<cv::Rect> >(k_values.size()));
            for (int d = 0; d < img_segmentations.size(); ++d)
                for (int s = 0; s < k_values.size(); ++s)
                    Segmentation::getBoundingBoxes(img_segmentations[d][s], img_regions[d][s]);

            Segmentation::calculateSignificantRegions(img_regions, regions_significant);

            t2 = std::chrono::high_resolution_clock::now();

            duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            std::cout << std::endl << "Segmentation Time for: " << file_name << ": " << duration / 1000 << "s" << std::endl;

            DisplayImg(img, "img");
            ShowManyImages("domains", img_domains, 2, 3);

            showSegmentations(img_domains[disp_domain], img_segmentations[disp_domain], img_regions[disp_domain]);

            int orgininal_region_cnt = 0;
            cv::Mat all_regions = img_domains[0].clone();
            for (const std::vector< std::vector<cv::Rect> > &a : img_regions) {
                for (const std::vector<cv::Rect> &b : a) {
                    orgininal_region_cnt += b.size();
                    for (const cv::Rect &c : b)
                        DrawBoundingBox(all_regions, c, cv::Scalar(255, 0, 0), false);
                }
            }

            std::cout << "detected regions: " << orgininal_region_cnt << std::endl;
            std::cout << "regions surviving: " << regions_significant.size() << std::endl;

            DisplayImg(all_regions, "all_region_proposals");
            showRegions(img_domains[0], regions_significant);
        }

        key = cv::waitKey();
    } while (key != 'q' && key != 'c' && key != 27);
    cv::destroyAllWindows();

    return 0;
}