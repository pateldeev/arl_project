#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/segmentation.hpp>

#include <iostream>

namespace Segmentation {

    void process(const cv::Mat &img, std::vector<cv::Rect> &proposals, std::vector<float> &scores);
    void showSegmentationResults(const cv::Mat &img, const std::vector<cv::Rect> &proposals, const std::vector<float> &scores, const std::string &window_name = "segmentation_results");

    struct RegionProposal;

    //6 Domains [BGR, HSV, LAB, I, RGI, YUV]
    void getDomains(const cv::Mat &img, std::vector<cv::Mat> &img_domains, int resize_h = 200);
    void getSegmentations(const std::vector<cv::Mat> &img_domains, std::vector< std::vector<cv::Mat> > &segmentations, const std::vector<float> &k_vals, float sigma = 0.8);
    void getBoundingBoxes(const cv::Mat &img_seg, std::vector<cv::Rect> &boxes, float max_region_size = 0.5);

    void generateRegionProposals(const std::vector< std::vector<cv::Mat> > &segmentations, std::vector<RegionProposal> &proposals, float max_region_size = 0.5);
    void mergeProposalsWithinSegmentationLevel(std::vector<RegionProposal> &proposals, float IOU_thresh = 0.92);
    void mergeProposalsBetweenSegmentationLevels(std::vector<RegionProposal> &proposals, float min_score = 1.f, float IOU_thresh = 0.95);

    //old methodology
    void calculateSignificantRegions(const std::vector< std::vector< std::vector<cv::Rect> > > &regions_img, std::vector< std::pair<cv::Rect, float> > &regions_significant, float IOU_thresh = 0.9);

    struct RegionProposal {

        RegionProposal(const cv::Rect &region_box, float region_score, int region_domain, int segmentation_level) : box(region_box), score(region_score), domain(region_domain), seg_level(segmentation_level), status(1) {
        }

        cv::Rect box;
        float score;
        int domain;
        int seg_level;
        int status; //1->valid & unmerged; 2->valid & merged; 0->invalid

        //returns true is successful

        bool tryMerge(RegionProposal &other, float IOU_thresh) {
            if (float((box & other.box).area()) / (box | other.box).area() >= IOU_thresh) {
                box |= other.box;
                score += other.score;
                other.status = 0;
                status = 2;
                return true; //merged
            }
            return false; //not merged
        }

        bool isValid(void) const {
            return status > 0;
        }

        bool containsMerged(void) const {
            return status == 2;
        }

        bool hasSegLevel(void) const {
            return seg_level >= 0;
        }

        friend std::ostream& operator<<(std::ostream &os, const RegionProposal &p) {
            os << p.seg_level << ',' << p.domain << '|' << p.score << '|' << p.box << "||" << p.status;
            return os;
        }
    };
};

#endif
