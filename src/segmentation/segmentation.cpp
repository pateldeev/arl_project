#include "segmentation.h"

#include "functions.h"

#include <opencv2/ximgproc/segmentation.hpp>

namespace Segmentation {

    float RegionProposal::total_merge_scores = 0;
    int RegionProposal::img_area = 0;

    void process(const cv::Mat &img, std::vector<cv::Rect> &proposals, std::vector<float> &scores) {
        std::vector<cv::Mat> img_domains;
        const int resize_h = 200;
        const int resize_w = img.cols * resize_h / img.rows;
        getDomains(img, img_domains, resize_h);

        std::vector< std::vector<cv::Mat> > img_segmentations; //[domain][k-value]
        getSegmentations(img_domains, img_segmentations);

        std::vector<RegionProposal> img_proposals;
        generateRegionProposals(img_segmentations, img_proposals);
        RegionProposal::total_merge_scores = 0;

        mergeProposalsWithinSegmentationLevel(img_proposals); //merge within same segmentation domain - different domains

        mergeProposalsBetweenSegmentationLevels(img_proposals); //merge different segmentation levels

        mergeProposalsCommonThroughoutDomain(img_proposals); //merge proposals found in all segmentation levels of a single domain

        getSignificantMergedRegions(img_proposals, proposals, scores); //get only regions merged between segmentation levels

        resizeRegions(proposals, img.cols, img.rows, resize_w, resize_h);
    }

    cv::Mat showSegmentationResults(const cv::Mat &img, const std::vector<cv::Rect> &proposals, const std::vector<float> &scores, const std::string &window_name, unsigned int display_outline_thickness, bool show_scores) {
        cv::Mat disp_img = img.clone();
        //std::srand(std::time(NULL));
        GetRandomColor(true);
        if (show_scores) {
            unsigned int disp_position = 0;

            std::vector<std::pair<float, unsigned int>> regions_by_scores;

            for (unsigned int i = 0; i < proposals.size(); ++i)
                regions_by_scores.emplace_back(scores[i], i);

            std::sort(regions_by_scores.begin(), regions_by_scores.end(), std::greater<std::pair<float, unsigned int>>());

            for (const std::pair<float, unsigned int> &r : regions_by_scores) {
                cv::Scalar color = GetRandomColor();
                DrawBoundingBox(disp_img, proposals[r.second], color, false, display_outline_thickness);
                std::string score("000.000");
                std::snprintf(const_cast<char*> (score.c_str()), score.size(), "%.3f", r.first);
                score.resize(score.find_first_of('\0'));
                WriteText(disp_img, score, 0.7, color, cv::Point(2, 20 * (++disp_position)));
            }
        } else {
            for (const cv::Rect r : proposals)
                DrawBoundingBox(disp_img, r, GetRandomColor(), false, display_outline_thickness);
        }

        DisplayImg(disp_img, window_name);
        return disp_img;
    }

    void getDomains(const cv::Mat &img, std::vector<cv::Mat> &img_domains, int resize_h) {
        const int resize_w = img.cols * resize_h / img.rows;
        RegionProposal::img_area = resize_h * resize_w;

        img_domains.clear();
        img_domains.resize(6); //6 Domains [BGR, HSV, LAB, I, RGI, YCrCb]

        //BGR
        cv::resize(img, img_domains[0], cv::Size(resize_w, resize_h));

        //HSV
        cv::cvtColor(img_domains[0], img_domains[1], cv::COLOR_BGR2HSV);

        //LAB
        cv::cvtColor(img_domains[0], img_domains[2], cv::COLOR_BGR2Lab);

        //I - intensity
        cv::cvtColor(img_domains[0], img_domains[3], cv::COLOR_BGR2GRAY);

        //RGI
        cv::Mat img_channels[3];
        cv::split(img_domains[0], img_channels);
        cv::Mat rgi_channels[3] = {img_channels[2], img_channels[1], img_domains[3]};
        cv::merge(rgi_channels, 3, img_domains[4]);

        //YCrCb
        cv::cvtColor(img_domains[0], img_domains[5], cv::COLOR_BGR2YCrCb);
    }

    void getSegmentations(const std::vector<cv::Mat> &img_domains, std::vector< std::vector<cv::Mat> > &segmentations, unsigned int seg_levels) {
        const float k_base = 600, k_step = 100;
        const std::vector <float> sigma_vals = {0.8, 0.8, 0.5, 0.5, 0.8, 0.5};
        segmentations.resize(img_domains.size(), std::vector<cv::Mat>(seg_levels));

        cv::Ptr<cv::ximgproc::segmentation::GraphSegmentation> gs = cv::ximgproc::segmentation::createGraphSegmentation();

        for (unsigned int i_domain = 0; i_domain < img_domains.size(); ++i_domain) {
            gs->setSigma(sigma_vals[i_domain]);

            float k = k_base;
            for (unsigned int i_seg = 0; i_seg < seg_levels; ++i_seg, k += k_step) {
                gs->setK(k);
                gs->processImage(img_domains[i_domain], segmentations[i_domain][i_seg]);

                if (i_seg == 0) {
                    double min, max;
                    cv::minMaxLoc(segmentations[i_domain][i_seg], &min, &max);
                    if (max < 10) {
                        while (max < 10 && k > k_step) {
                            k -= k_step;
                            gs->setK(k);
                            gs->processImage(img_domains[i_domain], segmentations[i_domain][i_seg]);
                            cv::minMaxLoc(segmentations[i_domain][i_seg], &min, &max);
                        }
                    } else if (max >= 20) {
                        while (max >= 20) {
                            k += k_step;
                            gs->setK(k);
                            gs->processImage(img_domains[i_domain], segmentations[i_domain][i_seg]);
                            cv::minMaxLoc(segmentations[i_domain][i_seg], &min, &max);
                        }
                    }
                }
            }
        }
    }

    void getBoundingBoxes(const cv::Mat &img_seg, std::vector<cv::Rect> &boxes, float max_region_size) {
        double min, max;
        cv::minMaxLoc(img_seg, &min, &max);
        int num_segs = max + 1;

        const float side_ignore_size = 3; //ignore any regions that touch boundaries of image
        const int max_points = max_region_size * img_seg.rows * img_seg.cols;

        std::unordered_map< int, std::vector<cv::Point> > seg_points_map;
        for (unsigned int i = 0; i < num_segs; ++i)
            seg_points_map[i]; //create entry for every segment in map

        const uint32_t * ptr_seg;
        std::unordered_map< int, std::vector<cv::Point> >::iterator seg_points;

        for (unsigned int i = 0; i < img_seg.rows; ++i) {
            ptr_seg = img_seg.ptr<uint32_t>(i);

            for (int j = 0; j < img_seg.cols; ++j) {
                int seg = ptr_seg[j];
                seg_points = seg_points_map.find(seg);
                if (seg_points != seg_points_map.end()) { //only concern ourselves with 
                    if (i < side_ignore_size || i > img_seg.rows - side_ignore_size || j < side_ignore_size || j > img_seg.cols - side_ignore_size) {
                        seg_points_map.erase(seg_points); //segment is near the edge - erase from map
                    } else {
                        if (seg_points->second.size() > max_points)
                            seg_points_map.erase(seg_points); //segment has too many points - erase from map
                        else
                            seg_points->second.push_back(cv::Point(j, i)); //add point to corresponding region
                    }
                }
            }
        }

        //create bounding rectangle based on points in map
        boxes.clear();
        for (const std::pair< int, std::vector<cv::Point> > &p : seg_points_map) {
            cv::Rect box = cv::boundingRect(p.second);
            //only predict region if is bigger than a minimal dimensional size and contains enough salient points

            if (box.width > img_seg.rows * 0.02 && box.height > img_seg.cols * 0.02 && float(p.second.size()) / box.area() > 0.15)
                boxes.emplace_back(box);
        }
    }

    void generateRegionProposals(const std::vector< std::vector<cv::Mat> > &segmentations, std::vector<RegionProposal> &proposals, float max_region_size) {
        proposals.clear();

        std::vector<float> segmentation_weights(segmentations[0].size());
        for (int w = 0; w < segmentation_weights.size(); ++w)
            segmentation_weights[w] = 1 / (1 + std::exp(-float(w + 1) / 4)); //weight per segmentation level.

        double min, max;
        int num_segs;
        const float side_ignore_size = 3; //size to ignore any regions that touch boundaries of image

        for (unsigned int d = 0; d < segmentations.size(); ++d) { //domain
            for (unsigned int s = 0; s < segmentations[d].size(); ++s) { //segmentation
                const cv::Mat img_seg = segmentations[d][s];

                cv::minMaxLoc(img_seg, &min, &max);
                num_segs = max + 1;

                const int max_points = max_region_size * img_seg.rows * img_seg.cols;

                std::unordered_map< int, std::vector<cv::Point> > seg_points_map;
                for (int i = 0; i < num_segs; ++i)
                    seg_points_map[i]; //create entry for every segment in map

                const uint32_t * ptr_seg;
                std::unordered_map< int, std::vector<cv::Point> >::iterator seg_points;

                for (unsigned int i = 0; i < img_seg.rows; ++i) {
                    ptr_seg = img_seg.ptr<uint32_t>(i);

                    for (unsigned int j = 0; j < img_seg.cols; ++j) {
                        int seg = ptr_seg[j];
                        seg_points = seg_points_map.find(seg);
                        if (seg_points != seg_points_map.end()) { //only concern ourselves with valid regions
                            if (i < side_ignore_size || i > img_seg.rows - side_ignore_size || j < side_ignore_size || j > img_seg.cols - side_ignore_size) {
                                seg_points_map.erase(seg_points); //segment is near the edge - erase from map
                            } else {
                                if (seg_points->second.size() > max_points)
                                    seg_points_map.erase(seg_points); //segment has too many points - erase from map
                                else
                                    seg_points->second.push_back(cv::Point(j, i)); //add point to corresponding region
                            }
                        }
                    }
                }

                //create bounding rectangle based on points in map
                for (const std::pair< int, std::vector<cv::Point> > &p : seg_points_map) {
                    cv::Rect box = cv::boundingRect(p.second);
                    //only predict region if is bigger than a minimal dimensional size and contains enough salient points

                    if (box.width > img_seg.rows * 0.02 && box.height > img_seg.cols * 0.02 && float(p.second.size()) / box.area() > 0.15)
                        proposals.emplace_back(box, segmentation_weights[s], d, s);
                }
            }
        }
    }

    void mergeProposalsWithinSegmentationLevel(std::vector<RegionProposal> &proposals, float IOU_thresh, float IU_diff_percentage) {
        if (proposals.size() <= 1)
            return;

        //sort by segmentation level (low to high) and area (low to high)
        RegionProposal::sortProposals(proposals, [](const RegionProposal & r1, const RegionProposal & r2)-> bool {
            return (r1.seg_level == r2.seg_level) ? (r1.box.area() < r2.box.area()) : (r1.seg_level < r2.seg_level);
        });

        std::vector<RegionProposal>::iterator seg_start = proposals.begin(), seg_end;
        for (int s_level = 0; s_level <= proposals.back().seg_level; ++s_level) {
            //get to start of proposals
            while (seg_start->seg_level != s_level)
                ++seg_start;

            //get to end of proposals
            seg_end = seg_start;
            if (s_level == proposals.back().seg_level)
                seg_end = proposals.end();
            else
                while ((++seg_end)->seg_level == s_level);

            //merge proposals in same level as s_level
            bool merged = true;
            while (merged) {
                merged = false;

                for (std::vector<RegionProposal>::iterator seg_candidate1 = seg_start; seg_candidate1 != seg_end - 1; ++seg_candidate1) {
                    if (!seg_candidate1->isValid())
                        continue; //ignore already merged regions

                    int max_candidate2_area = (seg_candidate1->box.area() / IOU_thresh)*1.25; //add extra cushion for possibility of rectangles out of order due to merging
                    for (std::vector<RegionProposal>::iterator seg_candidate2 = seg_candidate1 + 1; seg_candidate2 != seg_end; ++seg_candidate2) {
                        if (!seg_candidate2->isValid())
                            continue; //ignore already merged regions

                        if (seg_candidate2->box.area() < max_candidate2_area) {
                            if (seg_candidate1->tryMerge(*seg_candidate2, IOU_thresh, IU_diff_percentage)) {
                                seg_candidate1->domain = -1; //new merged belongs to no single domain
                                merged = true;
                                break; //merged - move onto next one
                            }
                        } else {
                            break; //reached largest possible rectangle that can be merged move on
                        }
                    }
                }
            }
        }

        RegionProposal::removeInvalidProposals(proposals); //remove merged regions
    }

    void mergeProposalsBetweenSegmentationLevels(std::vector<RegionProposal> &proposals, float min_score, float IOU_thresh, float IU_diff_percentage) {
        if (proposals.size() <= 1)
            return;

        //sort by segmentation level (low to high) and score (high to low)
        RegionProposal::sortProposals(proposals, [](const RegionProposal & r1, const RegionProposal & r2)-> bool {
            return (r1.seg_level == r2.seg_level) ? (r2.score < r1.score) : (r1.seg_level < r2.seg_level);
        });

        int num_seg_levels = proposals.back().seg_level + 1; //since proposals are sorted
        std::vector<RegionProposal>::iterator segmentation_cutoffs[num_seg_levels + 1]; //store where each segmentation starts

        segmentation_cutoffs[0] = proposals.begin();
        segmentation_cutoffs[num_seg_levels] = proposals.end(); //store extra iterator to end of vector for simplicity
        for (unsigned int i = 1; i < num_seg_levels; ++i) {
            segmentation_cutoffs[i] = segmentation_cutoffs[i - 1];
            while (segmentation_cutoffs[i] != proposals.end() && segmentation_cutoffs[i]->seg_level < i)
                ++segmentation_cutoffs[i]; //store location where next segmentations can be found in proposals
        }

        for (unsigned int i = 0; i < num_seg_levels; ++i) {
            //go through each significant region that could be merged
            for (std::vector<RegionProposal>::iterator region_to_merge = segmentation_cutoffs[i]; region_to_merge != segmentation_cutoffs[i + 1]; ++region_to_merge) {
                if (region_to_merge->score < min_score)
                    break; //reached lowest score in segmentation level that we will consider for merging - move onto next level
                if (!region_to_merge->isValid())
                    continue; //region has already been merged

                for (unsigned int j = 0; j < num_seg_levels; ++j) {
                    if (j == i)
                        continue; //skip checking regions in same segmentation level

                    //go through all other regions in segmentation level and try to merge
                    for (std::vector<RegionProposal>::iterator merge_candidate = segmentation_cutoffs[j]; merge_candidate != segmentation_cutoffs[j + 1]; ++merge_candidate) {
                        if (!merge_candidate->isValid() || !merge_candidate->hasSegLevel())
                            continue; //regions has already been merged

                        if (region_to_merge->tryMerge(*merge_candidate, IOU_thresh, IU_diff_percentage)) {
                            region_to_merge->seg_level = -1; //indicate merged with different segmentation level
                        }
                    }
                }
            }
        }

        RegionProposal::removeInvalidProposals(proposals); //remove merged regions
    }

    void mergeProposalsCommonThroughoutDomain(std::vector<RegionProposal> &proposals, float IOU_thresh, float IU_diff_percentage, unsigned int num_segmentations_levels) {
        if (proposals.size() <= 1)
            return;

        //sort by domain (low to high) and then segmentation level (low to high) and score
        RegionProposal::sortProposals(proposals, [](const RegionProposal & r1, const RegionProposal & r2)-> bool {
            return (r1.domain == r2.domain) ? (r1.seg_level < r2.seg_level) : (r1.domain < r2.domain);
        });

        std::vector<RegionProposal>::iterator d_start = proposals.begin(), d_end = proposals.begin();
        int target_domain = 0;

        while (d_end != proposals.end()) {
            //find start and end of target domain
            while (d_start->domain < target_domain && d_start != proposals.end()) //get to first unmerged proposal in domain 0
                ++d_start;
            d_end = d_start;

            while (d_end->domain == target_domain && d_end != proposals.end())
                ++d_end;

            mergeProposalsCommonInDomain(d_start, d_end, IOU_thresh, IU_diff_percentage, num_segmentations_levels); //merge proposals in domain using helper function

            ++target_domain; //go onto next domain
        }

        RegionProposal::removeInvalidProposals(proposals); //remove merged regions
    }

    //get only the proposals that have been merged between domains or found in all segmentations levels of one domain

    void getSignificantMergedRegions(const std::vector<RegionProposal> &proposals, std::vector<cv::Rect> &signficiant_regions, std::vector<float> &sigificant_region_scores) {
        //transfer the most significant proposals
        signficiant_regions.clear();
        sigificant_region_scores.clear();
        for (const RegionProposal &p : proposals) {
            if (!p.hasSegLevel()) {
                signficiant_regions.push_back(p.box);
                sigificant_region_scores.push_back(p.score / RegionProposal::total_merge_scores);
            }
        }
    }

    void resizeRegions(std::vector<cv::Rect> &regions, int original_w, int original_h, int resize_w, int resize_h) {
        //rescale rectangles
        int x, y, w, h;
        for (cv::Rect &r : regions) {
            x = original_w * (double(r.x) / resize_w);
            y = original_h * (double(r.y) / resize_h);
            w = original_w * (double(r.width) / resize_w);
            h = original_h * (double(r.height) / resize_h);

            r = cv::Rect(x, y, w, h);
        }
    }

    //helper function for mergeProposalsCommonThroughoutDomain
    //all regions between domain_start and domain_end must belong to same domain - should be sorted by segmentation_level

    void mergeProposalsCommonInDomain(std::vector<RegionProposal>::iterator domain_start, std::vector<RegionProposal>::iterator domain_end, float IOU_thresh, float IU_diff_percentage, unsigned int num_segmentations_levels) {
        std::vector<RegionProposal>::iterator seg_level_start[num_segmentations_levels + 1];
        int seg_level = -1;
        for (std::vector<RegionProposal>::iterator p = domain_start; p != domain_end; ++p) {
            if (p->seg_level == seg_level + 1)
                seg_level_start[++seg_level] = p;
        }
        seg_level_start[num_segmentations_levels] = domain_end;

        if (seg_level == num_segmentations_levels - 1) { //all segmentation levels have regions
            for (std::vector<RegionProposal>::iterator p = seg_level_start[0]; p != seg_level_start[1]; ++p) {
                RegionProposal * matching_regions[num_segmentations_levels];
                unsigned int i = 1;
                for (; i < num_segmentations_levels; ++i) {
                    matching_regions[i] = findMatchingProposal(p, seg_level_start[i], seg_level_start[i + 1], IOU_thresh, IU_diff_percentage);
                    if (!matching_regions[i]) //no matching region in segmentation level
                        break;
                }
                if (i == num_segmentations_levels) { //there is matching region in every every segmentation level
                    p->seg_level = -2;
                    p->status = 2;
                    for (unsigned int j = 1; j < num_segmentations_levels; ++j) {

                        p->score += matching_regions[j]->score;
                        matching_regions[j]->status = 0;
                    }
                    RegionProposal::total_merge_scores += p->score;
                }
            }
        }
    }


    //helper function for mergeProposalsCommonInDomain
    //returns pointer to proposal that matches given one in specified range

    RegionProposal* findMatchingProposal(std::vector<RegionProposal>::iterator proposal, std::vector<RegionProposal>::iterator seg_level_start, std::vector<RegionProposal>::iterator seg_level_end, float IOU_thresh, float IU_diff_percentage) {
        cv::Point proposal_center = 0.5 * (proposal->box.tl() + proposal->box.br());

        for (std::vector<RegionProposal>::iterator p = seg_level_start; p != seg_level_end; ++p) {
            if (p->box.contains(proposal_center)) { //only check for matching if box contains center
                float I = (proposal->box & p->box).area();
                float U = (proposal->box | p->box).area();
                if (I / U >= IOU_thresh || (U - I) <= IU_diff_percentage * RegionProposal::img_area)
                    return &(*p); //found matching
            }
        }
        return nullptr; //no matching found
    }

};