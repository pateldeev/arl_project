#include "functions.h"

#include "segmentation.h"

#include <iostream>
#include <chrono>
#include <opencv2/highgui.hpp>

void showSegmentations(const cv::Mat &img, const std::vector< std::vector<cv::Mat> > &segmentations, const std::vector< Segmentation::RegionProposal > &proposals, int disp_domain) {
    std::vector<cv::Mat> disp_segs = {img.clone()}, disp_regions_img = {img.clone()}, disp_regions_segs = {img.clone()}, disp_per_segmentation_level{img.clone()};

    int total_segmentation_cnt = 0;
    double min, max;
    for (const cv::Mat &s : segmentations[disp_domain]) {
        cv::minMaxLoc(s, &min, &max);
        total_segmentation_cnt += int(max);
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
    WriteText(disp_segs[0], std::to_string(total_segmentation_cnt));
    WriteText(disp_regions_img[0], std::to_string(total_region_count));
    WriteText(disp_regions_segs[0], std::to_string(total_region_count));

    DisplayMultipleImages("segmentations", disp_segs, 2, 3);
    DisplayMultipleImages("proposals_over_img", disp_regions_img, 2, 3);
    DisplayMultipleImages("proposals_over_seg", disp_regions_segs, 2, 3);
    DisplayMultipleImages("proposals_per_level", disp_per_segmentation_level, 2, 3);
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
        if (p.hasMerged()) {
            cv::Scalar color(std::rand() % 255, std::rand() % 255, std::rand() % 255);
            DrawBoundingBox(dips_regions_merged[p.seg_level + 1], p.box, color);
            std::string score("00.000");
            std::snprintf(const_cast<char*> (score.c_str()), score.size(), "%.2f", p.score);
            score.resize(score.find_first_of('\0'));
            WriteText(dips_regions_merged[0], score, 0.35, color, cv::Point(2 + 42 * p.seg_level, 10 * (++score_text_starting_pos[p.seg_level])));
        }
        ++region_cnt[p.seg_level];
    }

    int total_region_count = 0;
    for (unsigned int i = 0; i < region_cnt.size(); ++i) {
        WriteText(dips_regions_all[i + 1], std::to_string(region_cnt[i]));
        total_region_count += region_cnt[i];
    }
    WriteText(dips_regions_all[0], std::to_string(total_region_count));

    DisplayMultipleImages("per_segmentation_level_all", dips_regions_all, 2, 3);
    DisplayMultipleImages("filtered_per_level_merged", dips_regions_merged, 2, 3);
}

void showProposalsMergedWithinDomain(const std::vector<cv::Mat> &img_domain, const std::vector< Segmentation::RegionProposal > &proposals, const std::string &window_name = "Found_In_All_Domain_Levels") {
    std::vector<cv::Mat> disp_imgs(img_domain.size());
    for (unsigned int i = 0; i < disp_imgs.size(); ++i)
        disp_imgs[i] = img_domain[i].clone();

    for (const Segmentation::RegionProposal &p : proposals) {
        if (p.seg_level == -2)
            DrawBoundingBox(disp_imgs[p.domain], p.box, cv::Scalar(255, 0, 0));
    }

    DisplayMultipleImages(window_name, disp_imgs, 2, img_domain.size() / 2);
}

void showProposals(const cv::Mat &img, const std::vector<Segmentation::RegionProposal> &proposals, const std::string &window_name = "proposals") {
    cv::Mat disp_img = img.clone();

    for (const Segmentation::RegionProposal &p : proposals)
        DrawBoundingBox(disp_img, p.box, cv::Scalar(std::rand() % 255, std::rand() % 255, std::rand() % 255));

    WriteText(disp_img, std::to_string(proposals.size()));
    DisplayImg(disp_img, window_name);
}

void showSegmentationsAll(const std::vector<cv::Mat> &img_domains, const std::vector< std::vector<cv::Mat> > &segmentations, const std::vector< Segmentation::RegionProposal > &proposals, const std::string &window_name = "all_segmentations_in_one") {
    std::vector<cv::Mat> disp;

    for (unsigned int i = 0; i < img_domains.size(); ++i) {
        disp.push_back(img_domains[i].clone());

        if (disp.back().type() == CV_8UC1)
            cv::cvtColor(disp.back(), disp.back(), cv::COLOR_GRAY2BGR);

        for (const cv::Mat &seg : segmentations[i])
            disp.push_back(GetGraphSegmentationViewable(seg));
    }

    std::vector<unsigned int> regions_cnt(disp.size(), 0);
    for (const Segmentation::RegionProposal &p : proposals) {
        unsigned int domain_index = p.domain * (1 + segmentations[0].size());
        ++regions_cnt[domain_index];
        unsigned int seg_index = domain_index + p.seg_level + 1;
        ++regions_cnt[seg_index];
        DrawBoundingBox(disp[seg_index], p.box, cv::Scalar(0, 0, 0), false, 2);
    }

    for (unsigned int i = 0; i < regions_cnt.size(); ++i)
        WriteText(disp[i], std::to_string(regions_cnt[i]));

    unsigned int h = img_domains.size() * img_domains[0].rows;
    unsigned int w = (1 + segmentations[0].size()) * img_domains[0].cols;
    DisplayMultipleImages(window_name, disp, img_domains.size(), 1 + segmentations[0].size(), cv::Size(w, h));
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
        file_current = 10;
        file_max = 36;
    } else if (data_set == '2') {
        strcpy(file_name_format, "/home/dp/Downloads/data_mine/%d.png");
        file_current = 4;
        file_max = 20;
    } else {
        strcpy(file_name_format, "/home/dp/Downloads/data_cumulative/%d.png");
        file_max = 17;
    }

    std::vector<cv::Mat> img_domains;
    std::vector< std::vector<cv::Mat> > img_segmentations; //[domain][k-value]
    std::vector<Segmentation::RegionProposal> img_proposals;
    std::vector<Segmentation::RegionProposal> proposals_merged_within_levels, proposals_merged_between_levels, proposals_merged_common_in_domain;
    std::vector<cv::Rect> final_proposals;
    std::vector<float> final_proposal_scores;
    int disp_domain = 0;

    std::chrono::high_resolution_clock::time_point t1, t2;
    double duration;

    int key = 0;
    do {
        if (key == 'n')
            ++file_current %= (file_max + 1);
        else if (key == 'b' && --file_current < 0)
            file_current = file_max;
        else if (key == '+' || key == '=' || key == 171)
            ++disp_domain %= img_domains.size();
        else if ((key == '-' || key == 173) && --disp_domain < 0)
            disp_domain += img_domains.size();

        if (key == '+' || key == '=' || key == '-' || key == 171 || key == 173) {
            showSegmentations(img_domains[disp_domain], img_segmentations, img_proposals, disp_domain);
        } else {
            sprintf(file_name, file_name_format, file_current);
            img = cv::imread(file_name);

            img_proposals.clear();

            t1 = std::chrono::high_resolution_clock::now();

            Segmentation::getDomains(img, img_domains);
            Segmentation::getSegmentations(img_domains, img_segmentations);

            Segmentation::generateRegionProposals(img_segmentations, img_proposals);

            proposals_merged_within_levels = img_proposals;
            Segmentation::mergeProposalsWithinSegmentationLevel(proposals_merged_within_levels);

            proposals_merged_between_levels = proposals_merged_within_levels;
            Segmentation::mergeProposalsBetweenSegmentationLevels(proposals_merged_between_levels);

            proposals_merged_common_in_domain = proposals_merged_between_levels;
            Segmentation::mergeProposalsCommonThroughoutDomain(proposals_merged_common_in_domain);

            Segmentation::getSignificantMergedRegions(proposals_merged_common_in_domain, final_proposals, final_proposal_scores);

            t2 = std::chrono::high_resolution_clock::now();

            duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            std::cout << std::endl << "Segmentation Time for: " << file_name << ": " << duration / 1000 << "s" << std::endl;

            DisplayMultipleImages("domains", img_domains, 2, 3);

            showSegmentations(img_domains[disp_domain], img_segmentations, img_proposals, disp_domain);
            showProposalsMergedWithin(img_domains[0], proposals_merged_within_levels);
            showProposalsMergedWithinDomain(img_domains, proposals_merged_common_in_domain);
            showProposals(img_domains[0], img_proposals, "all_regions");

            Segmentation::resizeRegions(final_proposals, img.cols, img.rows, img.cols * 200 / img.rows, 200);
            Segmentation::showSegmentationResults(img, final_proposals, final_proposal_scores, "FINAL-Significant regions");

            showSegmentationsAll(img_domains, img_segmentations, img_proposals);
            std::cout << "total number of detected regions: " << img_proposals.size() << std::endl;
        }

        key = cv::waitKey();
    } while (key != 'q' && key != 'c' && key != 27);
    cv::destroyAllWindows();

    return 0;
}