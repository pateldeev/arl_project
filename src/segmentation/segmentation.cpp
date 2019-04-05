#include "segmentation.h"

#include "functions.h"

#include <opencv2/ximgproc/segmentation.hpp>

namespace Segmentation {

    float RegionProposal::total_merge_scores = 0;

    void process(const cv::Mat &img, std::vector<cv::Rect> &proposals, std::vector<float> &scores) {
        std::vector<cv::Mat> img_domains;
        const int resize_h = 200;
        const int resize_w = img.cols * resize_h / img.rows;
        getDomains(img, img_domains, resize_h);

        const std::vector<float> k_values = {600, 700, 800, 900, 1000};
        std::vector< std::vector<cv::Mat> > img_segmentations; //[domain][k-value]
        getSegmentations(img_domains, img_segmentations, k_values);

        std::vector<RegionProposal> img_proposals;
        generateRegionProposals(img_segmentations, img_proposals);
        RegionProposal::total_merge_scores = 0;

        mergeProposalsWithinSegmentationLevel(img_proposals); //merge per segmentation - within domain

        mergeProposalsBetweenSegmentationLevels(img_proposals); //merge different segmentation levels

        getSignificantMergedRegions(img_proposals, proposals, scores); //get only regions merged between segmentation levels

        resizeRegions(proposals, img.cols, img.rows, resize_w, resize_h);
    }

    void showSegmentationResults(const cv::Mat &img, const std::vector<cv::Rect> &proposals, const std::vector<float> &scores, const std::string &window_name, unsigned int box_thickness) {
        cv::Mat disp_img = img.clone();
        int disp_position = 0;

        for (unsigned int i = 0; i < proposals.size(); ++i) {
            cv::Scalar color(std::rand() % 255, std::rand() % 255, std::rand() % 255);
            DrawBoundingBox(disp_img, proposals[i], color, false, box_thickness);
            std::string score("000.000");
            std::snprintf(const_cast<char*> (score.c_str()), score.size(), "%.3f", scores[i]);
            score.resize(score.find_first_of('\0'));
            WriteText(disp_img, score, 0.7, color, cv::Point(2, 20 * (++disp_position)));
        }

        DisplayImg(disp_img, window_name);
    }

    void getDomains(const cv::Mat &img, std::vector<cv::Mat> &img_domains, int resize_h) {
        const int resize_w = img.cols * resize_h / img.rows;

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

    void getSegmentations(const std::vector<cv::Mat> &img_domains, std::vector< std::vector<cv::Mat> > &segmentations, const std::vector<float> &k_vals, float sigma) {
        segmentations.resize(img_domains.size(), std::vector<cv::Mat>(k_vals.size()));
        cv::Ptr<cv::ximgproc::segmentation::GraphSegmentation> gs = cv::ximgproc::segmentation::createGraphSegmentation(sigma);

        for (unsigned int i_k = 0; i_k < k_vals.size(); ++i_k) {
            gs->setK(k_vals[i_k]);
            for (unsigned int i_d = 0; i_d < img_domains.size(); ++i_d)
                gs->processImage(img_domains[i_d], segmentations[i_d][i_k]);
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

    //proposals will be sorted by segmentation level and area (low to high)

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

        //sort by segmentation level (low to hight) and area (low to high)
        std::sort(proposals.begin(), proposals.end(), [](const RegionProposal &r1, const RegionProposal & r2)-> bool {
            if (r1.seg_level == r2.seg_level)
                return r1.box.area() < r2.box.area();
            else
                return r1.seg_level < r2.seg_level;
        });
    }

    //proposals must be sorted by segmentation level and area (low to high)
    //will return proposals sorted by segmentation level (low to high) and score (high to low)

    void mergeProposalsWithinSegmentationLevel(std::vector<RegionProposal> &proposals, float IOU_thresh) {
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
                            if (seg_candidate1->tryMerge(*seg_candidate2, IOU_thresh)) {
                                merged = true;
                                break; //segments merged
                            }
                        } else {
                            break; //reached largest possible rectangle that can be merged move on
                        }
                    }
                }
            }
        }

        //remove merged regions
        proposals.erase(std::remove_if(proposals.begin(), proposals.end(), [](const RegionProposal & p)->bool {
            return !p.isValid();
        }), proposals.end());

        //sort by segmentation level (low to high) and score (high to low)
        std::sort(proposals.begin(), proposals.end(), [](const RegionProposal &r1, const RegionProposal & r2)-> bool {
            if (r1.seg_level == r2.seg_level)
                return r2.score < r1.score;
            else
                return r1.seg_level < r2.seg_level;
        });
    }

    //proposals must be sorted by segmentation level (low to high) and score (high to low)
    //will return with intra-domain merged proposals (seg_level == -1) at front sorted by score (high to low)

    void mergeProposalsBetweenSegmentationLevels(std::vector<RegionProposal> &proposals, float min_score, float IOU_thresh) {
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

                        if (region_to_merge->tryMerge(*merge_candidate, IOU_thresh)) {
                            region_to_merge->seg_level = -1; //indicate merged with different segmentation level
                        }
                    }
                }
            }
        }

        //sort by segmentation level (low to high) and score (high to low) with newly merged regions at front
        std::sort(proposals.begin(), proposals.end(), [](const RegionProposal &r1, const RegionProposal & r2)-> bool {
            if (r1.seg_level == r2.seg_level)
                return r2.score < r1.score;
            else
                return r1.seg_level < r2.seg_level;
        });
    }

    //get only the proposals that have been merged between domains - proposals must be sorted by segmentation level (low to high)

    void getSignificantMergedRegions(const std::vector<RegionProposal> &proposals, std::vector<cv::Rect> &signficiant_regions, std::vector<float> &sigificant_region_scores) {
        //transfer the most significant proposals
        signficiant_regions.clear();
        sigificant_region_scores.clear();
        for (const RegionProposal &p : proposals) {
            if (p.seg_level == -1) {
                signficiant_regions.push_back(p.box);
                sigificant_region_scores.push_back(p.score / RegionProposal::total_merge_scores);
            } else {
                break;
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



    //old methodology - for selectiveSegmentationMain

    void calculateSignificantRegions(const std::vector< std::vector< std::vector<cv::Rect> > > &regions_img, std::vector< std::pair<cv::Rect, float> > &regions_significant, float IOU_tresh) {
        regions_significant.clear();

        std::vector<float> weights(regions_img[0].size());
        for (int i = 0; i < weights.size(); ++i)
            weights[i] = 1 / (1 + std::exp(-i)); //weight per segmentation level.

        for (const std::vector< std::vector<cv::Rect> > &regions_domain : regions_img) { //iterate through each domain
            for (unsigned int s = 0; s < regions_domain.size(); ++s) {
                for (const cv::Rect &region : regions_domain[s]) {
                    bool merged = false;

                    for (std::pair<cv::Rect, float> &r_s : regions_significant) {
                        if (float((region & r_s.first).area()) / (region | r_s.first).area() >= IOU_tresh) {
                            r_s.first = region | r_s.first;
                            r_s.second += weights[s];
                            merged = true;
                        }
                    }

                    if (!merged)
                        regions_significant.emplace_back(region, weights[s]);
                }
            }
        }

        int first_valid = 0;
        bool merged = true;
        while (merged) {
            merged = false;
            for (unsigned int i = first_valid; i < regions_significant.size() - 1; ++i) {
                for (unsigned int j = i + 1; j < regions_significant.size(); ++j) {
                    if (float((regions_significant[i].first & regions_significant[j].first).area()) / (regions_significant[i].first | regions_significant[j].first).area() >= IOU_tresh) {

                        regions_significant[j].first |= regions_significant[i].first;
                        regions_significant[j].second += regions_significant[i].second;

                        std::iter_swap(regions_significant.begin() + first_valid, regions_significant.begin() + i);

                        ++first_valid;
                        merged = true;
                        break;
                    }
                }
            }
        }

        regions_significant.erase(regions_significant.begin(), regions_significant.begin() + first_valid);

        //sort regions by likelyhood of being and object
        std::sort(regions_significant.begin(), regions_significant.end(), [](const std::pair<cv::Rect, float> &r1, const std::pair<cv::Rect, float> &r2)-> bool {
            return r1.second < r2.second;
        });
    }

};