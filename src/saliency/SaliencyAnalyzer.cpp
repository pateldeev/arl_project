#include "saliency.h"
#include "functions.h"

#include <opencv2/saliency/saliencyBaseClasses.hpp>
#include <opencv2/saliency/saliencySpecializedClasses.hpp>
#include <queue>
#include <opencv2/core.hpp>

namespace SaliencyFilter {

    SaliencyAnalyzer::SaliencyAnalyzer(const cv::Mat &img) : m_saliency_mean(-1), m_saliency_std(-1) {
        cv::saliency::StaticSaliencySpectralResidual::create()->computeSaliency(img, m_saliency_map);

        cv::Scalar saliency_mean, saliency_std;
        cv::meanStdDev(m_saliency_map, saliency_mean, saliency_std);
        m_saliency_mean = saliency_mean[0];
        m_saliency_std = saliency_std[0];

        std::cout << std::endl << "saliency mean|std_dev: " << m_saliency_mean << "|" << m_saliency_std << std::endl;
    }

    SaliencyAnalyzer::~SaliencyAnalyzer(void) {

    }

    void SaliencyAnalyzer::addSegmentedRegion(const cv::Rect &region, float segmentation_score) {
        m_regions.emplace_back(m_saliency_map, region, segmentation_score, m_saliency_std);
    }

    void SaliencyAnalyzer::mergeSubRegions(float overlap_thresh, float min_size_of_inner) {
        sortRegionsByArea();

        //go through and try to merge each region with any that are smaller than it
        for (int i = 0; i < m_regions.size() - 1; ++i) {
            if (m_regions[i].status != 1)
                continue; //only consider valid regions

            for (int j = i + 1; j < m_regions.size(); ++j) {
                if (m_regions[j].status != 1)
                    continue; //only consider valid regions

                //if there is enough area overlap, merge regions
                cv::Rect overlap = m_regions[i].box & m_regions[j].box;
                float overlap_num_pixels = (overlap.height + 1) * (overlap.width + 1);
                if (overlap_num_pixels > min_size_of_inner * m_regions[i].box_num_pixels && overlap_num_pixels > overlap_thresh * m_regions[j].box_num_pixels) {

                    float sal_overlap = cv::sum(GetSubRegionOfMat(m_saliency_map, overlap))[0];
                    float avg_saliency_between = (m_regions[i].box_sal - sal_overlap) / (m_regions[i].box_num_pixels - overlap_num_pixels);
                    float saliency_change = (m_regions[j].avg_sal - avg_saliency_between) / m_saliency_std;

                    if (saliency_change > m_regions[i].std_change_from_surroundings) { //keep smaller
                        m_regions[j].score += m_regions[i].score;
                        m_regions[i].status = 0;
                        break;
                    } else { //keep bigger
                        m_regions[i].score += m_regions[j].score;
                        m_regions[j].status = 0;
                    }
                }
            }
        }
        removeInvalidRegions();
    }

    void SaliencyAnalyzer::applySaliencyThreshold(void) {
        float region_score_expected = 0.5 / m_regions.size();
        for (Region &r : m_regions) {
            float score_percent = r.score - region_score_expected;
            r.ensureRegionSaliency(1.25 - 2 * score_percent, m_saliency_mean, m_saliency_std);
        }
        removeInvalidRegions();
    }

    void SaliencyAnalyzer::computeSaliencyDescriptors(void) {
        for (Region &r : m_regions) {
            cv::Mat disp = m_img.clone();
            DrawBoundingBox(disp, r.box, cv::Scalar(0, 255, 0));
            r.computeDescriptorsAlongAllEdges(m_saliency_map, disp);
        }
    }

    void SaliencyAnalyzer::getRegionsSurviving(std::vector<cv::Rect> &regions) const {
        regions.clear();
        for (const Region &r : m_regions) {

            if (r.status == 1)
                regions.emplace_back(r.box);
        }
    }

    void SaliencyAnalyzer::sortRegionsByArea(void) {
        std::sort(m_regions.begin(), m_regions.end(), [](const Region &r1, const Region & r2)->bool {
            return r2.box_num_pixels < r1.box_num_pixels;
        });
    }

    void SaliencyAnalyzer::removeInvalidRegions(void) {
        m_regions.erase(std::remove_if(m_regions.begin(), m_regions.end(), [](const Region & r)->bool {
            return r.status != 1;
        }), m_regions.end());
    }

    SaliencyAnalyzer::Region::Region(const cv::Mat &saliency_map, const cv::Rect &region, float score, float saliency_map_std) : box(region), box_num_pixels((region.width + 1) * (region.height + 1)), status(1), score(score) {
        cv::Scalar box_mean, box_std;
        cv::meanStdDev(GetSubRegionOfMat(saliency_map, box), box_mean, box_std);
        avg_sal = box_mean[0];
        std_sal = box_std[0];
        box_sal = avg_sal * box_num_pixels;

        //ensure region stands out from immediate surroundings (x2 size)- will bring back if region stands out a lot
        cv::Point doubled_tl(std::max(0, box.tl().x - (box.width / 2)), std::max(0, box.tl().y - (box.height / 2)));
        cv::Point doubled_br(std::min(saliency_map.cols - 1, box.br().x + (box.width / 2)), std::min(saliency_map.rows - 1, box.br().y + (box.height / 2)));
        box_double = cv::Rect(doubled_tl, doubled_br);
        box_num_pixels_double = (box_double.width + 1) * (box_double.height + 1);

        cv::meanStdDev(GetSubRegionOfMat(saliency_map, box_double), box_mean, box_std);
        avg_sal_double = box_mean[0];
        std_sal_double = box_std[0];
        box_sal_double = avg_sal_double * box_num_pixels_double;
        avg_sal_surroundings = (box_sal_double - box_sal) / (box_num_pixels_double - box_num_pixels);
        std_change_from_surroundings = (avg_sal - avg_sal_surroundings) / saliency_map_std;

        std_saliency_map = saliency_map_std;
    }

    void SaliencyAnalyzer::Region::ensureRegionSaliency(float std_thresh_overall, float saliency_mean, float saliency_std) {
        //ensure region is salient enough
        if (status == 1 && (((avg_sal - saliency_mean) / saliency_std) < std_thresh_overall)) {
            status = -1; //remove if not salient compared to entire image
            std::cout << "Removed in comparison to ENTIRE" << std::endl;
        }

        //ensure region stands out from immediate surroundings (x2 size)- will bring back if region stands out a lot
        if (status == 1 && avg_sal_surroundings > 0.8 * avg_sal) {
            status = -2; //remove if not salient compared to surroundings
            std::cout << "Removed in comparison to DOUBLE" << std::endl;
        } else if (status == -1 && avg_sal > 2.5 * avg_sal_surroundings) {
            status = 1; //restore if very salient compared to surroundings
            std::cout << "RESTORED" << std::endl;
        }
    }

    void SaliencyAnalyzer::Region::computeDescriptorsAlongAllEdges(const cv::Mat &saliency_map, const cv::Mat &img) {
        RECT_SIDES edges_order[4];
        int optimal_changes[4];

        if (box.width > box.height) { //change left/right first than up/down
            edges_order[0] = RECT_SIDES::LEFT;
            edges_order[1] = RECT_SIDES::RIGHT;
            edges_order[2] = RECT_SIDES::TOP;
            edges_order[3] = RECT_SIDES::BOTTOM;
        } else { //change up/down first than left/right
            edges_order[0] = RECT_SIDES::TOP;
            edges_order[1] = RECT_SIDES::BOTTOM;
            edges_order[2] = RECT_SIDES::LEFT;
            edges_order[3] = RECT_SIDES::RIGHT;
        }

        for (unsigned int i = 0; i < 2; ++i) {
            computeDescriptorAlongSingleEdge(edges_order[i], saliency_map);

            std::cout << std::endl << "Calling visualization |" << avg_sal << "|" << std_sal << "| Surrounding |" << avg_sal_double << "|" << std_sal_double << std::endl;
            edges[edges_order[i]].visualizeDescriptor(img);
            //CV_Assert(cv::waitKey() != 'q');

            optimal_changes[edges_order[i]] = edges[edges_order[i]].computeOptimalChange();
            resizeBoxEdge(edges_order[i], optimal_changes[edges_order[i]]);
        }

        for (unsigned int i = 2; i < 4; ++i) {
            computeDescriptorAlongSingleEdge(edges_order[i], saliency_map);

            std::cout << std::endl << "Calling visualization |" << avg_sal << "|" << std_sal << "| Surrounding |" << avg_sal_double << "|" << std_sal_double << std::endl;
            edges[edges_order[i]].visualizeDescriptor(img);
            //CV_Assert(cv::waitKey() != 'q');

            optimal_changes[edges_order[i]] = edges[edges_order[i]].computeOptimalChange();
            resizeBoxEdge(edges_order[i], optimal_changes[edges_order[i]]);

        }

        //return;
        cv::Mat disp = img.clone();
        DrawBoundingBox(disp, box, cv::Scalar(0, 0, 255));
        DisplayImg(disp, "CHANGED");
        //CV_Assert(cv::waitKey() != 'q');
    }

    void SaliencyAnalyzer::Region::computeDescriptorAlongSingleEdge(RECT_SIDES side, const cv::Mat &saliency_map) {
        const float max_expansion = 0.3;

        if (side == RECT_SIDES::LEFT)
            edges[side].compute(saliency_map, false, box.tl(), box.height, box.width, box.width * max_expansion, box.width * max_expansion);
        else if (side == RECT_SIDES::RIGHT)
            edges[side].compute(saliency_map, false, cv::Point(box.tl().x + box.width, box.tl().y), box.height, -box.width, box.width * max_expansion, box.width * max_expansion);
        else if (side == RECT_SIDES::TOP)
            edges[side].compute(saliency_map, true, box.tl(), box.width, box.height, box.height * max_expansion, box.height * max_expansion);
        else if (side == RECT_SIDES::BOTTOM)
            edges[side].compute(saliency_map, true, cv::Point(box.tl().x, box.tl().y + box.height), box.width, -box.height, box.height * max_expansion, box.height * max_expansion);
    }

    void SaliencyAnalyzer::Region::resizeBoxEdge(RECT_SIDES side, int change) {
        if (side == RECT_SIDES::LEFT)
            box = cv::Rect(box.tl() + cv::Point(change, 0), box.br());
        else if (side == RECT_SIDES::RIGHT)
            box = cv::Rect(box.tl(), box.br() + cv::Point(change, 0));
        else if (side == RECT_SIDES::TOP)
            box = cv::Rect(box.tl() + cv::Point(0, change), box.br());
        else if (side == RECT_SIDES::BOTTOM)
            box = cv::Rect(box.tl(), box.br() + cv::Point(0, change));
    }
};