#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/opencv.hpp>

namespace Segmentation {

    void process(const cv::Mat &img, std::vector<cv::Rect> &proposals, std::vector<float> &scores);
    cv::Mat showSegmentationResults(const cv::Mat &img, const std::vector<cv::Rect> &proposals, const std::vector<float> &scores, const std::string &window_name = "segmentation_results", unsigned int display_outline_thickness = 1, bool show_scores = true);

    struct RegionProposal;

    //6 Domains [BGR, HSV, LAB, I, RGI, YUV]
    void getDomains(const cv::Mat &img, std::vector<cv::Mat> &img_domains, int resize_h = 200);
    void getSegmentations(const std::vector<cv::Mat> &img_domains, std::vector< std::vector<cv::Mat> > &segmentations, unsigned int seg_levels = 5);
    void getBoundingBoxes(const cv::Mat &img_seg, std::vector<cv::Rect> &boxes, float max_region_size = 0.5);

    void generateRegionProposals(const std::vector< std::vector<cv::Mat> > &segmentations, std::vector<RegionProposal> &proposals, float max_region_size = 0.5);
    void mergeProposalsWithinSegmentationLevel(std::vector<RegionProposal> &proposals, float IOU_thresh = 0.95, float IU_diff_percentage = 0.005);
    void mergeProposalsBetweenSegmentationLevels(std::vector<RegionProposal> &proposals, float min_score = 1.f, float IOU_thresh = 0.95, float IU_diff_percentage = 0.005);
    void mergeProposalsCommonThroughoutDomain(std::vector<RegionProposal> &proposals, float IOU_thresh = 0.95, float IU_diff_percentage = 0.0075, unsigned int num_segmentations_levels = 5);
    void getSignificantMergedRegions(const std::vector<RegionProposal> &proposals, std::vector<cv::Rect> &signficiant_regions, std::vector<float> &sigificant_region_scores);
    void resizeRegion(cv::Rect &region, int original_w, int original_h, int resize_w, int resize_h);
    void resizeRegions(std::vector<cv::Rect> &regions, int original_w, int original_h, int resize_w, int resize_h);

    struct RegionProposal {

        RegionProposal(const cv::Rect &region_box, float region_score, int region_domain, int segmentation_level) : box(region_box), score(region_score), domain(region_domain), seg_level(segmentation_level), status(1) {
        }

        cv::Rect box;
        float score;
        int domain;
        int seg_level; //-1->between segmentation and domain; -2->only between segmentation
        int status; //1->valid & unmerged; 2->valid & merged; 0->invalid

        //returns true is successful

        bool tryMerge(RegionProposal &other, float IOU_thresh, float IU_diff_percentage) {
            float I = (box & other.box).area();
            float U = (box | other.box).area();
            if (I / U >= IOU_thresh || (U - I) <= IU_diff_percentage * RegionProposal::img_area) { //merge successful  
                box |= other.box;
                score += other.score;

                //keep track of total score of all merges
                if (other.status == 1)
                    total_merge_scores += other.score;
                if (status == 1)
                    total_merge_scores += score;

                //update status showing merge
                other.status = 0;
                status = 2;

                return true; //merged
            }
            return false; //not merged
        }

        bool isValid(void) const {
            return status > 0;
        }

        bool hasMerged(void) const {
            return status == 2;
        }

        bool hasSegLevel(void) const {
            return seg_level >= 0;
        }

        friend std::ostream& operator<<(std::ostream &os, const RegionProposal &p) {
            os << p.seg_level << ',' << p.domain << '|' << p.score << '|' << p.box << "||" << p.status;
            return os;
        }

        static float total_merge_scores;
        static int img_area;

        static void removeInvalidProposals(std::vector<RegionProposal> &proposals) {
            proposals.erase(std::remove_if(proposals.begin(), proposals.end(), [](const RegionProposal & p)->bool {
                return !p.isValid();
            }), proposals.end());
        }

        static void sortProposals(std::vector<RegionProposal> &proposals, const std::function<bool (const RegionProposal&, const RegionProposal&) > &comparator) {
            std::sort(proposals.begin(), proposals.end(), comparator);
        }
    };

    //helper functions

    void mergeProposalsCommonInDomain(std::vector<RegionProposal>::iterator domain_start, std::vector<RegionProposal>::iterator domain_end, float IOU_thresh, float IU_diff_percentage, unsigned int num_segmentations_levels);

    RegionProposal* findMatchingProposal(std::vector<RegionProposal>::iterator proposal, std::vector<RegionProposal>::iterator seg_level_start, std::vector<RegionProposal>::iterator seg_level_end, float IOU_thresh, float IU_diff_percentage);

};

#endif
