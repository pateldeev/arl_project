#include "saliency.h"
#include "functions.h"

#include <opencv2/saliency/saliencyBaseClasses.hpp>
#include <opencv2/saliency/saliencySpecializedClasses.hpp>

namespace SaliencyFilter {

    SaliencyAnalyzer::SaliencyAnalyzer(const cv::Mat &img) {
        m_saliency_map_equalized = computeSaliencyMap(img, true);
        m_saliency_map_unequalized = computeSaliencyMap(img, false);

        m_img = img.clone();

        cv::Scalar saliency_mean, saliency_std;
        cv::meanStdDev(m_saliency_map_equalized, saliency_mean, saliency_std);
        Region::saliency_map_mean = saliency_mean[0];
        Region::saliency_map_std = saliency_std[0];

        std::cout << std::endl << "saliency equalized mean|std_dev: " << saliency_mean[0] << "|" << saliency_std[0] << std::endl;
        cv::meanStdDev(m_saliency_map_unequalized, saliency_mean, saliency_std);
        std::cout << std::endl << "saliency unequalized mean|std_dev: " << saliency_mean[0] << "|" << saliency_std[0] << std::endl;

    }

    SaliencyAnalyzer::~SaliencyAnalyzer(void) {

    }

    void SaliencyAnalyzer::addSegmentedRegion(const cv::Rect &region, float segmentation_score) {
        m_regions.emplace_back(m_saliency_map_equalized, region, segmentation_score);
    }

    void SaliencyAnalyzer::tryMergingSubRegions(float force_merge_threshold) {
        Region::sortRegionsByArea(m_regions);

        for (unsigned int i = 0; i < m_regions.size() - 1; ++i) {
            if (m_regions[i].status != 1)
                continue; //only consider valid regions

            for (unsigned int j = i + 1; j < m_regions.size(); ++j) {
                if (m_regions[j].status != 1)
                    continue; //only consider valid regions

                //consider removal if smaller is contained by bigger
                if ((m_regions[i].box & m_regions[j].box).area() >= m_regions[j].box.area() * 0.95) {
                    //std::cout << "Considering Merge" << std::endl;
                    if (m_regions[j].box_num_pixels > force_merge_threshold * m_regions[i].box_num_pixels)
                        m_regions[i].forceMergeWithSubRegion(m_regions[j]);
                    else
                        m_regions[i].tryMergeWithSubRegion(m_regions[j]);

#if 0
                    if (m_regions[i].status == -3 || m_regions[j].status == -3) {
                        cv::Mat disp = m_img.clone();
                        cv::Mat disp2 = m_img.clone();
                        DrawBoundingBox(disp, m_regions[i].box, cv::Scalar(0, 0, 255), false, 3);
                        DrawBoundingBox(disp, m_regions[j].box, cv::Scalar(255, 0, 0), false, 3);
                        DrawBoundingBox(disp2, m_regions[j].status == -3 ? m_regions[i].box : m_regions[j].box, cv::Scalar(0, 255, 0), false, 3);
                        DisplayImg(disp, "MERGED");
                        if (cv::waitKey() == 's') {
                            SaveImg(disp(m_regions[j].box_double), "/home/dp/Downloads/poster/saliency/merge_candidates.png");
                            SaveImg(disp2(m_regions[j].status == -3 ? m_regions[i].box_double : m_regions[j].box_double), "/home/dp/Downloads/poster/saliency/merge_outcome.png");
                        }
                    }
#endif
                    if (m_regions[i].status == -3)
                        break;
                }
            }
        }
        Region::removeInvalidRegions(m_regions);
    }

    void SaliencyAnalyzer::ensureSaliencyOfRegions(void) {
        float region_score_expected = 0.5 / m_regions.size();
        for (Region &r : m_regions) {
            r.ensureRegionSaliency(r.score - region_score_expected);

#if 0
            if (r.status != 1) {
                std::vector<cv::Mat> disp;
                disp.push_back(m_img.clone());
                disp.push_back(m_saliency_map_unequalized.clone());
                cv::normalize(disp.back(), disp.back(), 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::cvtColor(disp.back(), disp.back(), cv::COLOR_GRAY2BGR);
                DrawBoundingBox(disp[0], r.box, cv::Scalar(0, 0, 255), false, 3);
                DrawBoundingBox(disp[1], r.box, cv::Scalar(0, 0, 255), false, 3);
                disp.push_back(disp[0](r.box_double).clone());
                disp.push_back(disp[1](r.box_double).clone());

                DisplayMultipleImages("Region Removed", disp, 1, disp.size());
                if (cv::waitKey() == 's')
                    SaveImg(disp[3], "/home/dp/Downloads/poster/saliency/rejected.png");
            } else {
                std::vector<cv::Mat> disp;
                disp.push_back(m_img.clone());
                disp.push_back(m_saliency_map_unequalized.clone());
                cv::normalize(disp.back(), disp.back(), 0, 255, cv::NORM_MINMAX, CV_8UC1);
                cv::cvtColor(disp.back(), disp.back(), cv::COLOR_GRAY2BGR);
                DrawBoundingBox(disp[0], r.box, cv::Scalar(0, 255, 0), false, 3);
                DrawBoundingBox(disp[1], r.box, cv::Scalar(0, 255, 0), false, 3);
                disp.push_back(disp[0](r.box_double).clone());
                disp.push_back(disp[1](r.box_double).clone());

                DisplayMultipleImages("Region Accepted", disp, 1, disp.size());
                if (cv::waitKey() == 's')
                    SaveImg(disp[3], "/home/dp/Downloads/poster/saliency/accepted.png");
            }
#endif
        }
        Region::removeInvalidRegions(m_regions);
    }

    bool SaliencyAnalyzer::reconcileOverlappingRegions(float overlap_thresh) {
        if (m_regions.size() <= 1)
            return false;

        Region::sortRegionsByArea(m_regions);
        bool regions_reconciled = false;

        for (unsigned int i = 0; i < m_regions.size() - 1; ++i) {
            if (m_regions[i].status != 1)
                continue; //only consider valid regions

            for (unsigned int j = i + 1; j < m_regions.size(); ++j) {
                if (m_regions[j].status != 1)
                    continue; //only consider valid regions

                //consider removal if smaller has significant intersection with bigger one
                cv::Rect I(m_regions[i].box & m_regions[j].box);
                if (I != m_regions[j].box && I.area() > overlap_thresh * m_regions[j].box.area()) {
                    if (m_regions[i].score > 3 * m_regions[j].score) {
                        m_regions[j].status = -5;
                        m_regions[i].score += 0.5 * m_regions[j].score;
                    } else if (m_regions[j].score > 3 * m_regions[i].score) {
                        m_regions[i].status = -5;
                        m_regions[j].score += 0.5 * m_regions[i].score;
                    } else {
#if 0
                        std::vector<cv::Mat> disp = {m_img.clone(), m_img.clone()};
                        DrawBoundingBox(disp[0], m_regions[i].box, cv::Scalar(255, 0, 0));
                        WriteText(disp[0], std::to_string(m_regions[i].score), 1.0, cv::Scalar(255, 0, 0), cv::Point(10, 30));
                        DrawBoundingBox(disp[0], m_regions[j].box, cv::Scalar(0, 255, 0));
                        WriteText(disp[0], std::to_string(m_regions[j].score), 1.0, cv::Scalar(0, 255, 0), cv::Point(10, 70));
#endif                       
                        m_regions[i] = Region::computeReconciledRegion(m_regions[i], m_regions[j], m_saliency_map_equalized, m_saliency_map_unequalized, m_img);
                        m_regions[j].status = -5;
#if 0
                        DrawBoundingBox(disp[1], m_regions[i].box, cv::Scalar(0, 0, 255));
                        DisplayMultipleImages("combining|new", disp, 1, 2);
                        cv::waitKey();
#endif
                    }
                    regions_reconciled = true;
                }
            }
        }
        if (regions_reconciled)
            Region::removeInvalidRegions(m_regions);

        return regions_reconciled;
    }

    void SaliencyAnalyzer::ensureDistinguishabilityOfRegions(void) {
        if (m_regions.size() <= 1)
            return;

        cv::Mat img_gray;
        cv::cvtColor(m_img, img_gray, cv::COLOR_BGR2GRAY);

        for (Region &r : m_regions) {
            bool keep = true;

            cv::Point box_size_change(r.box.width / 4, r.box.height / 4);
            cv::Point box_outer_tl = r.box.tl() - box_size_change;
            cv::Point box_outer_br = r.box.br() + box_size_change;
            box_outer_tl.x = std::max(0, box_outer_tl.x);
            box_outer_tl.y = std::max(0, box_outer_tl.y);
            box_outer_br.x = std::min(m_img.cols - 1, box_outer_br.x);
            box_outer_br.y = std::min(m_img.rows - 1, box_outer_br.y);
            cv::Rect box_inner(r.box.tl() + box_size_change, r.box.br() - box_size_change);
            cv::Rect box_outer(box_outer_tl, box_outer_br);

            float box_entropy_inner = CalcSpatialEntropy(img_gray, box_inner);
            float box_entropy = CalcSpatialEntropy(img_gray, r.box);
            float box_entropy_outer = CalcSpatialEntropy(img_gray, box_outer);
            float box_entropy_around = CalcSpatialEntropy(img_gray, box_outer, r.box);

            float entropy_change_inner_and_box = box_entropy_inner / box_entropy;
            float entropy_change_box_and_outer = box_entropy / box_entropy_outer;
            float entropy_change_inner_and_outer = box_entropy_inner / box_entropy_outer;
            float entropy_change_around = box_entropy_around / box_entropy;

            if (std::abs(entropy_change_inner_and_box - entropy_change_box_and_outer) < 0.12 && std::abs(entropy_change_inner_and_box - entropy_change_inner_and_outer) < 0.12 && std::abs(entropy_change_box_and_outer - entropy_change_inner_and_outer) < 0.12)
                keep = false;
#if 0
            if (!keep) {
                std::vector<cv::Mat> disp = {m_img.clone(), m_img.clone()};
                DrawBoundingBox(disp[0], r.box, cv::Scalar(255, 0, 0));
                WriteText(disp[0], std::to_string(r.score), 1.0, cv::Scalar(255, 0, 0), cv::Point(10, 30));

                cv::Scalar color;
                cv::Point pos(10, -10), change(0, 40);

                color = cv::Scalar(0, 255, 0);
                DrawBoundingBox(disp[1], box_inner, color);
                WriteText(disp[1], std::to_string(box_entropy_inner), 1.0, color, (pos += change));

                color = cv::Scalar(255, 0, 0);
                DrawBoundingBox(disp[1], r.box, color);
                WriteText(disp[1], std::to_string(box_entropy), 1.0, color, (pos += change));

                color = cv::Scalar(0, 0, 255);
                DrawBoundingBox(disp[1], box_outer, color);
                WriteText(disp[1], std::to_string(box_entropy_outer), 1.0, color, (pos += change));

                color = cv::Scalar(255, 0, 255);
                WriteText(disp[1], std::to_string(box_entropy_around), 1.0, color, (pos += change));

                color = cv::Scalar(0, 255, 0);
                WriteText(disp[1], std::to_string(entropy_change_inner_and_box), 1.0, color, (pos += change));
                color = cv::Scalar(255, 0, 0);
                WriteText(disp[1], std::to_string(entropy_change_box_and_outer), 1.0, color, (pos += change));
                color = cv::Scalar(0, 0, 255);
                WriteText(disp[1], std::to_string(entropy_change_inner_and_outer), 1.0, color, (pos += change));
                color = cv::Scalar(255, 0, 255);
                WriteText(disp[1], std::to_string(entropy_change_around), 1.0, color, (pos += change));

                color = cv::Scalar(255, 0, 255);
                WriteText(disp[1], keep ? "KEEP" : "REJECT", 1.0, color, (pos += change));
                DisplayMultipleImages("entropy_test", disp, 1, 2, cv::Size(1800, 700), true);
            }
#endif
            if (!keep)
                r.status = -6;
        }
        Region::removeInvalidRegions(m_regions);
    }

    void SaliencyAnalyzer::computeDescriptorsAndResizeRegions(void) {
        for (Region &r : m_regions) {
#if 0
            static int x = -1;
            if (++x == 1 || x == 5 || x == 7)
                continue;
            //if (x == 3)

            cv::Mat sal;
            cv::saliency::StaticSaliencySpectralResidual::create()->computeSaliency(m_img, sal);
            r.computeDescriptorsAndResize(sal, m_img);
#else
            r.computeDescriptorsAndResize(m_saliency_map_equalized, m_img);
#endif
        }

        cv::Scalar saliency_mean, saliency_std;
        cv::meanStdDev(m_saliency_map_equalized, saliency_mean, saliency_std);
        //Region::saliency_map_mean = saliency_mean[0];
        //Region::saliency_map_std = saliency_std[0];
        for (Region &r : m_regions)
            r = Region(m_saliency_map_equalized, r.box, r.score);
    }

    void SaliencyAnalyzer::keepBestRegions(unsigned int keep_num) {
        if (m_regions.size() <= keep_num)
            return;
        unsigned int remove = m_regions.size() - keep_num;
        std::priority_queue<std::pair<float, unsigned int>> region_entropies;

        cv::Mat img_gray;
        cv::cvtColor(m_img, img_gray, cv::COLOR_BGR2GRAY);
        for (unsigned int i = 0; i < m_regions.size(); ++i) {
            cv::Point box_size_change(m_regions[i].box.width / 4, m_regions[i].box.height / 4);
            cv::Point box_outer_tl = m_regions[i].box.tl() - box_size_change;
            cv::Point box_outer_br = m_regions[i].box.br() + box_size_change;
            box_outer_tl.x = std::max(0, box_outer_tl.x);
            box_outer_tl.y = std::max(0, box_outer_tl.y);
            box_outer_br.x = std::min(m_img.cols - 1, box_outer_br.x);
            box_outer_br.y = std::min(m_img.rows - 1, box_outer_br.y);
            cv::Rect box_inner(m_regions[i].box.tl() + box_size_change, m_regions[i].box.br() - box_size_change);
            cv::Rect box_outer(box_outer_tl, box_outer_br);

            float box_entropy_inner = CalcSpatialEntropy(img_gray, box_inner);
            float box_entropy = CalcSpatialEntropy(img_gray, m_regions[i].box);
            float box_entropy_outer = CalcSpatialEntropy(img_gray, box_outer);
            float box_entropy_around = CalcSpatialEntropy(img_gray, box_outer, m_regions[i].box);

            float entropy_change_inner_and_box = box_entropy_inner / box_entropy;
            float entropy_change_box_and_outer = box_entropy / box_entropy_outer;
            float entropy_change_inner_and_outer = box_entropy_inner / box_entropy_outer;
            float entropy_change_around = box_entropy_around / box_entropy;

            float total_diffs = std::abs(entropy_change_inner_and_box - entropy_change_box_and_outer) + std::abs(entropy_change_inner_and_box - entropy_change_inner_and_outer) + std::abs(entropy_change_box_and_outer - entropy_change_inner_and_outer);

#if 0
            std::vector<cv::Mat> disp = {m_img.clone(), m_img.clone()};
            DrawBoundingBox(disp[0], m_regions[i].box, cv::Scalar(255, 0, 0));
            WriteText(disp[0], std::to_string(m_regions[i].score), 1.0, cv::Scalar(255, 0, 0), cv::Point(10, 30));

            cv::Scalar color;
            cv::Point pos(10, -10), change(0, 40);

            color = cv::Scalar(0, 255, 0);
            DrawBoundingBox(disp[1], box_inner, color);
            WriteText(disp[1], std::to_string(box_entropy_inner), 1.0, color, (pos += change));

            color = cv::Scalar(255, 0, 0);
            DrawBoundingBox(disp[1], m_regions[i].box, color);
            WriteText(disp[1], std::to_string(box_entropy), 1.0, color, (pos += change));

            color = cv::Scalar(0, 0, 255);
            DrawBoundingBox(disp[1], box_outer, color);
            WriteText(disp[1], std::to_string(box_entropy_outer), 1.0, color, (pos += change));

            color = cv::Scalar(0, 0, 128);
            WriteText(disp[1], std::to_string(box_entropy_around), 1.0, color, (pos += change));

            color = cv::Scalar(0, 255, 0);
            WriteText(disp[1], std::to_string(entropy_change_inner_and_box), 1.0, color, (pos += change));
            color = cv::Scalar(255, 0, 0);
            WriteText(disp[1], std::to_string(entropy_change_box_and_outer), 1.0, color, (pos += change));
            color = cv::Scalar(0, 0, 255);
            WriteText(disp[1], std::to_string(entropy_change_inner_and_outer), 1.0, color, (pos += change));
            color = cv::Scalar(0, 0, 128);
            WriteText(disp[1], std::to_string(entropy_change_around), 1.0, color, (pos += change));

            color = cv::Scalar(255, 0, 255);
            WriteText(disp[1], std::to_string(total_diffs), 1.0, color, (pos += change));
            DisplayMultipleImages("entropy_test", disp, 1, 2, cv::Size(1800, 700), true);
#endif
            region_entropies.push(std::make_pair(total_diffs, i));
            //region_entropies.push(std::make_pair(-total_diffs, i)); //for 02, 09, 10
        }

#if 0
        for (unsigned int i = 0; i < m_regions.size(); ++i) {
            if (i == 1 || i == 2 || i == 5)
                continue;
            m_regions[i].status = -7;
        }
#else
        for (unsigned int i = 0; i < remove; ++i) {
            m_regions[region_entropies.top().second].status = -7;
            region_entropies.pop();
        }
#endif
        Region::removeInvalidRegions(m_regions);
    }

    void SaliencyAnalyzer::getRegionsSurviving(std::vector<cv::Rect> &regions) const {
        regions.clear();
        for (const Region &r : m_regions) {
            if (r.status == 1)
                regions.emplace_back(r.box);
        }
    }
};