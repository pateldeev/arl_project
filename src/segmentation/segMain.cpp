#include "functions.h"

#include "segmentation.h"

#include <iostream>
#include <chrono>
#include <opencv2/highgui.hpp>

void showSegmentations(const cv::Mat &img, const std::vector< std::vector<cv::Mat> > &segmentations, const std::vector< Segmentation::RegionProposal > &proposals, int disp_domain) {
    std::vector<cv::Mat> disp_segs = {img}, disp_regions_img = {img.clone()}, disp_regions_segs = {img.clone()}, disp_per_segmentation_level{img.clone()};

    for (const cv::Mat &s : segmentations[disp_domain]) {
        disp_segs.push_back(GetGraphSegmentationViewable(s, true));
        disp_regions_img.push_back(img.clone());
        disp_regions_segs.push_back(GetGraphSegmentationViewable(s));
        disp_per_segmentation_level.push_back(img.clone());
    }

    std::vector<int> region_cnt(segmentations[disp_domain].size());
    std::vector<int> region_cnt_segementation(segmentations[disp_domain].size());
    for (const Segmentation::RegionProposal &p : proposals) {
        if (p.domain == disp_domain) {
            DrawBoundingBox(disp_regions_img[p.seg_level + 1], p.box);
            DrawBoundingBox(disp_regions_segs[p.seg_level + 1], p.box);
            ++region_cnt[p.seg_level];
        }
        DrawBoundingBox(disp_per_segmentation_level[p.seg_level + 1], p.box);
        ++region_cnt_segementation[p.seg_level];
    }

    int total_region_count = 0;
    for (int i = 0; i < region_cnt.size(); ++i) {
        WriteText(disp_regions_img[i + 1], std::to_string(region_cnt[i]));
        WriteText(disp_regions_segs[i + 1], std::to_string(region_cnt[i]));
        WriteText(disp_per_segmentation_level[i + 1], std::to_string(region_cnt_segementation[i]));
        total_region_count += region_cnt[i];
    }
    WriteText(disp_regions_img[0], std::to_string(total_region_count));
    WriteText(disp_regions_segs[0], std::to_string(total_region_count));

    ShowManyImages("segmentations", disp_segs, 2, 3);
    ShowManyImages("proposals_over_img", disp_regions_img, 2, 3);
    ShowManyImages("proposals_over_seg", disp_regions_segs, 2, 3);
    ShowManyImages("proposals_per_level", disp_per_segmentation_level, 2, 3);
}

void showProposalsMergedWithin(const cv::Mat &img, const std::vector< Segmentation::RegionProposal > &proposals) {
    std::vector<cv::Mat> dips_regions_all = {img.clone()}, dips_regions_merged = {img.clone()};
    std::vector<int> region_cnt;
    std::vector<int> score_text_starting_pos;

    for (const Segmentation::RegionProposal &p : proposals) {

        while (dips_regions_all.size() <= p.seg_level + 1) {
            dips_regions_all.push_back(img.clone());
            region_cnt.push_back(0);
            dips_regions_merged.push_back(img.clone());
            score_text_starting_pos.push_back(0);
        }

        DrawBoundingBox(dips_regions_all[p.seg_level + 1], p.box, cv::Scalar(std::rand() % 255, std::rand() % 255, std::rand() % 255));
        if (p.containsMerged()) {
            cv::Scalar color(std::rand() % 255, std::rand() % 255, std::rand() % 255);
            DrawBoundingBox(dips_regions_merged[p.seg_level + 1], p.box, color);
            std::string score("00.000");
            std::snprintf(const_cast<char*> (score.c_str()), score.size(), "%.2f", p.score);
            score.resize(score.find_first_of('\0'));
            WriteText(dips_regions_merged[0], score, 0.4, color, cv::Point(2 + 42 * p.seg_level, 20 * (++score_text_starting_pos[p.seg_level])));
        }
        ++region_cnt[p.seg_level];
    }

    int total_region_count = 0;
    for (int i = 0; i < region_cnt.size(); ++i) {
        WriteText(dips_regions_all[i + 1], std::to_string(region_cnt[i]));
        total_region_count += region_cnt[i];
    }
    WriteText(dips_regions_all[0], std::to_string(total_region_count));

    ShowManyImages("filtered_per_level_all", dips_regions_all, 2, 3);
    ShowManyImages("filtered_per_level_merged", dips_regions_merged, 2, 3);
}

void showProposals(const cv::Mat &img, const std::vector<Segmentation::RegionProposal> &proposals, const std::string &window_name = "proposals", bool disp_scores = false) {
    cv::Mat disp_img = img.clone();
    int disp_position = 0;

    for (const Segmentation::RegionProposal &p : proposals) {
        cv::Scalar color(std::rand() % 255, std::rand() % 255, std::rand() % 255);
        DrawBoundingBox(disp_img, p.box, color);
        if (disp_scores) { //display score if requested
            std::string score("00.000");
            std::snprintf(const_cast<char*> (score.c_str()), score.size(), "%.2f", p.score);
            score.resize(score.find_first_of('\0'));
            WriteText(disp_img, score, 0.4, color, cv::Point(2, 20 * (++disp_position)));
        }
    }
    if (!disp_scores)
        WriteText(disp_img, std::to_string(proposals.size()));

    DisplayImg(disp_img, window_name);
}

int main(int argc, char * argv[]) {
#if 1
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

    const std::vector<float> k_values = {600, 700, 800, 900, 1000};

    std::vector<cv::Mat> img_domains;
    std::vector< std::vector<cv::Mat> > img_segmentations; //[domain][k-value]
    std::vector<Segmentation::RegionProposal> img_proposals;
    std::vector<Segmentation::RegionProposal> proposals_merged_within_levels, proposals_merged_between_levels;
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

        if (key == '+' || key == '=' || key == '-' || key == 171 || key == 173) {
            showSegmentations(img_domains[disp_domain], img_segmentations, img_proposals, disp_domain);
        } else {
            sprintf(file_name, file_name_format, file_count);
            img = cv::imread(file_name);
            //img = cv::imread("/home/dp/Downloads/IMG_0834.jpeg");

            img_proposals.clear();

            t1 = std::chrono::high_resolution_clock::now();

            Segmentation::getDomains(img, img_domains);
            Segmentation::getSegmentations(img_domains, img_segmentations, k_values);

            Segmentation::generateRegionProposals(img_segmentations, img_proposals);

            proposals_merged_within_levels = img_proposals;
            Segmentation::mergeProposalsWithinSegmentationLevel(proposals_merged_within_levels);
            proposals_merged_between_levels = proposals_merged_within_levels;
            Segmentation::mergeProposalsBetweenSegmentationLevels(proposals_merged_between_levels);

            t2 = std::chrono::high_resolution_clock::now();

            duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            std::cout << std::endl << "Segmentation Time for: " << file_name << ": " << duration / 1000 << "s" << std::endl;

            DisplayImg(img, "img");
            ShowManyImages("domains", img_domains, 2, 3);

            showSegmentations(img_domains[disp_domain], img_segmentations, img_proposals, disp_domain);
            showProposals(img_domains[0], img_proposals, "all_regions");
            showProposalsMergedWithin(img_domains[0], proposals_merged_within_levels);

            std::vector<Segmentation::RegionProposal> temp;
            for (const Segmentation::RegionProposal &p : proposals_merged_between_levels) {
                if (p.seg_level == -1)
                    temp.push_back(p);
                else
                    break;
            }
            showProposals(img_domains[0], temp, "merged_regions", true);

            std::cout << "total number of detected regions: " << img_proposals.size() << std::endl;
        }

        key = cv::waitKey();
    } while (key != 'q' && key != 'c' && key != 27);
    cv::destroyAllWindows();

    return 0;
}