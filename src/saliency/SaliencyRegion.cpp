#include "saliency.h"
#include "functions.h"

#include <opencv2/saliency/saliencyBaseClasses.hpp>
#include <opencv2/saliency/saliencySpecializedClasses.hpp>

namespace SaliencyFilter {

    float SaliencyAnalyzer::Region::saliency_map_mean = -1;
    float SaliencyAnalyzer::Region::saliency_map_std = -1;

    SaliencyAnalyzer::Region::Region(const cv::Mat &saliency_map, const cv::Rect &region, float score) : box(region), box_num_pixels((region.width + 1) * (region.height + 1)), status(1), score(score) {
        cv::Scalar box_mean, box_std;
        cv::meanStdDev(GetSubRegionOfMat(saliency_map, box), box_mean, box_std);
        avg_sal = box_mean[0];
        std_sal = box_std[0];
        box_sal = avg_sal * box_num_pixels;

        cv::Point doubled_tl(std::max(0, box.tl().x - (box.width / 2)), std::max(0, box.tl().y - (box.height / 2)));
        cv::Point doubled_br(std::min(saliency_map.cols - 1, box.br().x + (box.width / 2)), std::min(saliency_map.rows - 1, box.br().y + (box.height / 2)));
        box_double = cv::Rect(doubled_tl, doubled_br);
        box_num_pixels_double = (box_double.width + 1) * (box_double.height + 1);

        cv::meanStdDev(GetSubRegionOfMat(saliency_map, box_double), box_mean, box_std);
        avg_sal_double = box_mean[0];
        std_sal_double = box_std[0];
        box_sal_double = avg_sal_double * box_num_pixels_double;
        avg_sal_surroundings = (box_sal_double - box_sal) / (box_num_pixels_double - box_num_pixels);
        std_change_from_surroundings = (avg_sal - avg_sal_surroundings) / Region::saliency_map_std;
    }

    void SaliencyAnalyzer::Region::forceMergeWithSubRegion(Region &sub) {//sub must be subregion
        float avg_sal_between = (box_sal - sub.box_sal) / (box_num_pixels - sub.box_num_pixels);
        //std::cout << avg_sal_between << "|" << sub.avg_sal << std::endl;
        if (sub.avg_sal > avg_sal_between + 0.5 * Region::saliency_map_std) { //keep smaller
            //std::cout << "Keeping smaller" << std::endl;
            status = -3;
            sub.score += score / 2;
        } else { //keep larger
            //std::cout << "Keeping larger" << std::endl;
            sub.status = -3;
            score += sub.score / 2;
        }
    }

    void SaliencyAnalyzer::Region::tryMergeWithSubRegion(Region & sub) {//sub must be subregion
        float avg_sal_between = (box_sal - sub.box_sal) / (box_num_pixels - sub.box_num_pixels);
        if (avg_sal_between > sub.avg_sal) {
            //std::cout << "Removing smaller" << std::endl;
            sub.status = -3;
            score += sub.score / 2;
        } else {
            //std::cout << "Keeping both" << std::endl;
        }
    }

    void SaliencyAnalyzer::Region::ensureRegionSaliency(float std_thresh_overall) {
        //std::cout << avg_sal << "|" << avg_sal_double << '|' << avg_sal_surroundings << "||" << std_change_from_surroundings << "|||" << std_thresh_overall << std::endl;
        if (status == 1 && avg_sal < Region::saliency_map_mean)
            status = -1; //remove if not salient compared to entire image

        if (status == 1 && avg_sal < avg_sal_surroundings + 0.2 * Region::saliency_map_std)
            status = -2; //remove if not salient compared to surroundings
    }

    void SaliencyAnalyzer::Region::forceMergeWithRegion(Region &other, const cv::Rect &I, const cv::Mat &sal_map) {
        int I_num_pixels = (I.width + 1) * (I.height + 1);
        float I_sal = cv::sum(GetSubRegionOfMat(sal_map, I))[0];
        float I_avg_sal = I_sal / I_num_pixels;
        float this_avg_sal = (box_sal - I_sal) / (box_num_pixels - I_num_pixels);
        float other_avg_sal = (other.box_sal - I_sal) / (other.box_num_pixels - I_num_pixels);

        std::cout << I_avg_sal << "|" << this_avg_sal << "|" << other_avg_sal << std::endl;
        if (this_avg_sal > other_avg_sal) {
            other.status = -4;
            score += other.score / 2;
            std::cout << "Removing Other" << std::endl;
        } else {

            status = -4;
            other.score += score / 2;
            std::cout << "Removing This" << std::endl;
        }
    }

    void SaliencyAnalyzer::Region::computeDescriptorsAndResize(const cv::Mat &saliency_map, const cv::Mat &img) {
        cv::Mat disp_img = img.clone();
        DrawBoundingBox(disp_img, box, cv::Scalar(0, 255, 0));


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
#define DISPLAY_DESCRIPTORS 0
#if DISPLAY_DESCRIPTORS
        std::vector<cv::Mat> disp = {img.clone()};
        //cv::normalize(disp.back(), disp.back(), 0, 255, cv::NORM_MINMAX, CV_8UC1);
        //cv::cvtColor(disp.back(), disp.back(), cv::COLOR_GRAY2BGR);
        DrawBoundingBox(disp.back(), box, cv::Scalar(0, 0, 255), false, 3);
        disp.push_back(disp.back()(box_double));
#endif

        for (unsigned int i = 0; i < 2; ++i) { //move in one axis first
            computeDescriptorAlongEdge(edges_order[i], saliency_map);
#if DISPLAY_DESCRIPTORS
            std::cout << std::endl << "Calling visualization on " << edges_order[i] << " :" << avg_sal << " | " << std_sal << " | Surrounding | " << avg_sal_double << " | " << std_sal_double << std::endl;
            edges[edges_order[i]].visualizeDescriptor(disp_img);
            //CV_Assert(cv::waitKey() != 'q');
#endif
            optimal_changes[edges_order[i]] = edges[edges_order[i]].computeOptimalChange();

            resizeBoxEdge(edges_order[i], optimal_changes[edges_order[i]]); //resize to optimal
        }

        for (unsigned int i = 2; i < 4; ++i) {//move in other axis
            computeDescriptorAlongEdge(edges_order[i], saliency_map);
#if DISPLAY_DESCRIPTORS
            std::cout << std::endl << "Calling visualization |" << avg_sal << "|" << std_sal << "| Surrounding |" << avg_sal_double << "|" << std_sal_double << std::endl;
            edges[edges_order[i]].visualizeDescriptor(disp_img);
            //CV_Assert(cv::waitKey() != 'q');
#endif 
            optimal_changes[edges_order[i]] = edges[edges_order[i]].computeOptimalChange();

            resizeBoxEdge(edges_order[i], optimal_changes[edges_order[i]]); //resize to optimal

        }

#if DISPLAY_DESCRIPTORS   
        disp.push_back(img.clone());
        DrawBoundingBox(disp.back(), box, cv::Scalar(0, 255, 0), false, 3);
        disp.push_back(disp.back()(box_double));
        DisplayMultipleImages("CHANGED", disp, 1, disp.size());
        //CV_Assert(cv::waitKey() != 'q');
        if (cv::waitKey() == 's') {
            SaveImg(disp[1], "/home/dp/Downloads/poster/saliency/resize_before.png");
            SaveImg(disp[3], "/home/dp/Downloads/poster/saliency/resize_after.png");
        }
#endif
    }

    void SaliencyAnalyzer::Region::computeDescriptorAlongEdge(RECT_SIDES side, const cv::Mat &saliency_map) {
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

    SaliencyAnalyzer::Region SaliencyAnalyzer::Region::computeReconciledRegion(const Region &r1, const Region &r2, const cv::Mat &saliency_map_equalized, const cv::Mat &saliency_map_unequalized, const cv::Mat &img) {
        LineDescriptor rect_side;
        const cv::Point &r1_tl = r1.box.tl(), &r1_br = r1.box.br(), &r2_tl = r2.box.tl(), &r2_br = r2.box.br();
        cv::Point reconciled_tl, reconciled_br;

        //left
        if (std::abs(r1_tl.x - r2_tl.x) <= 2) {
            reconciled_tl.x = (r1_tl.x + r2_tl.x) / 2;
        } else {
            if (r1_tl.x < r2_tl.x)
                rect_side.compute(saliency_map_unequalized, false, r1_tl, r1.box.height, r1.box.width, 2 + r2_tl.x - r1_tl.x, 2);
            else
                rect_side.compute(saliency_map_unequalized, false, r1_tl, r1.box.height, r1.box.width, 2, 2 + r1_tl.x - r2_tl.x);
            reconciled_tl.x = r1_tl.x + rect_side.computeOptimalChange();
        }

        //right
        if (std::abs(r1_br.x - r2_br.x) <= 2) {
            reconciled_br.x = (r1_br.x + r2_br.x) / 2;
        } else {
            if (r1_br.x < r2_br.x)
                rect_side.compute(saliency_map_unequalized, false, cv::Point(r1_tl.x + r1.box.width, r1_tl.y), r1.box.height, -r1.box.width, 2 + r2_br.x - r1_br.x, 2);
            else
                rect_side.compute(saliency_map_unequalized, false, cv::Point(r1_tl.x + r1.box.width, r1_tl.y), r1.box.height, -r1.box.width, 2, 2 + r1_br.x - r2_br.x);
            reconciled_br.x = r1_br.x + rect_side.computeOptimalChange();
        }

        //top
        if (std::abs(r1_tl.y - r2_tl.y) <= 2) {
            reconciled_tl.y = (r1_tl.y + r2_tl.y) / 2;
        } else {
            if (r1_tl.y < r2_tl.y)
                rect_side.compute(saliency_map_unequalized, true, cv::Point(reconciled_tl.x, r1_tl.y), reconciled_br.x - reconciled_tl.x, r1.box.height, 2 + r2_tl.y - r1_tl.y, 2);
            else
                rect_side.compute(saliency_map_unequalized, true, cv::Point(reconciled_tl.x, r1_tl.y), reconciled_br.x - reconciled_tl.x, r1.box.height, 2, 2 + r1_tl.y - r2_tl.y);
            reconciled_tl.y = r1_tl.y + rect_side.computeOptimalChange();
        }

        //bottom
        if (std::abs(r1_br.y - r2_br.y) <= 2) {
            reconciled_br.y = (r1_br.y + r2_br.y) / 2;
        } else {
            if (r1_br.y < r2_br.y)
                rect_side.compute(saliency_map_unequalized, true, cv::Point(reconciled_tl.x, r1_tl.y + r1.box.height), reconciled_br.x - reconciled_tl.x, -r1.box.height, 2 + r2_br.y - r1_br.y, 2);

            else
                rect_side.compute(saliency_map_unequalized, true, cv::Point(reconciled_tl.x, r1_tl.y + r1.box.height), reconciled_br.x - reconciled_tl.x, -r1.box.height, 2, 2 + r1_br.y - r2_br.y);
            reconciled_br.y = r1_br.y + rect_side.computeOptimalChange();
        }

        return Region(saliency_map_equalized, cv::Rect(reconciled_tl, reconciled_br), 0.75 * (r1.score + r2.score));
    }

    void SaliencyAnalyzer::Region::sortRegionsByArea(std::vector<Region> &regions) {
        std::sort(regions.begin(), regions.end(), [](const Region & r1, const Region & r2)->bool {
            return r2.box_num_pixels < r1.box_num_pixels;
        });
    }

    void SaliencyAnalyzer::Region::removeInvalidRegions(std::vector<Region> &regions) {
        regions.erase(std::remove_if(regions.begin(), regions.end(), [](const Region & r)->bool {
            return r.status != 1;
        }), regions.end());
    }
};