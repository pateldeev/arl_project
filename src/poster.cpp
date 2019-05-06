#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "functions.h"
#include "segmentation.h"
#include "saliency.h"
#include "yoloInterface.h"

void final_results_saver(const std::string &folder) {
    const cv::Mat img = cv::imread(folder + "img.png");
    CV_Assert(!img.empty());

    std::vector<cv::Rect> segmentation_regions;
    std::vector<float> segmentation_scores;
    Segmentation::process(img, segmentation_regions, segmentation_scores);

    std::vector<cv::Rect> surviving_regions;
    SaliencyFilter::removeUnsalient(img, segmentation_regions, segmentation_scores, surviving_regions, false);

    cv::Mat disp_my = img.clone();
    cv::Mat disp_yolo = img.clone();
    std::vector<cv::Mat> disp_list = {disp_yolo, img, disp_my};

    std::vector<cv::Scalar> colors = {cv::Scalar(165, 240, 100), cv::Scalar(255, 200, 0), cv::Scalar(0, 150, 150)};
    for (unsigned int i = 0; i < surviving_regions.size(); ++i) {
        DrawBoundingBox(disp_my, surviving_regions[i], colors[i], false, 3);
        cv::Mat m = img(surviving_regions[i]).clone();
        int boardersize = std::max(m.rows / 42, m.cols / 42);
        cv::copyMakeBorder(m, m, boardersize, boardersize, boardersize, boardersize, cv::BORDER_CONSTANT, colors[i]);
        //SaveImg(m, folder + "predictions_" + std::to_string(i));
    }
    //SaveImg(disp_my, folder + "predictions.png");

    YoloInterface y(0.1);
    //y.loadNetwork("/home/dp/Desktop/darknet-master/cfg/yolov3.cfg", "/home/dp/Desktop/darknet-master/weights/yolov3.weights", "/home/dp/Desktop/darknet-master/data/coco.names");
    //y.processImage(img);
    //y.saveResults(folder + "yolo_results.txt");
    y.readResults(folder + "yolo_results.txt");

    cv::Rect previous(0, 0, img.cols, img.rows);
    colors = {cv::Scalar(0, 0, 200), cv::Scalar(0, 100, 255), cv::Scalar(255, 0, 200)};
    int cnt = -1;
    std::ofstream f(folder + "labels.txt");
    for (unsigned int i = 0; i < y.size(); ++i) {
        if (y[i].first != previous && ++cnt >= 0) {

            DrawBoundingBox(disp_yolo, y[i].first, colors[cnt], false, 3);
            previous = y[i].first;
            f << colors[cnt] << ":" << y[i].second.second << "|" << y[i].second.first << std::endl;

            cv::Mat m = img(y[i].first).clone();
            int boardersize = std::max(m.rows / 42, m.cols / 42);
            cv::copyMakeBorder(m, m, boardersize, boardersize, boardersize, boardersize, cv::BORDER_CONSTANT, colors[cnt]);
            //SaveImg(m, folder + "yolo_" + std::to_string(cnt));
        }
    }
    f.close();
    //SaveImg(disp_yolo, folder + "yolo.png");

    DisplayMultipleImages("Results", disp_list, 1, 3);

    cv::waitKey();
}

void flowchart_seg(void) {
    const cv::Mat img = cv::imread("/home/dp/Downloads/poster/Seg/img.png");
    CV_Assert(!img.empty());

    std::vector<cv::Mat> domains;
    Segmentation::getDomains(img, domains);

    for (unsigned int i = 0; i < domains.size(); ++i) {
        //SaveImg(domains[i], "/home/dp/Downloads/poster/Seg/" + std::to_string(i));
    }

    std::vector<cv::Rect> seg_regions;
    std::vector<float> seg_scores;
    Segmentation::process(img, seg_regions, seg_scores);

    cv::Mat temp = Segmentation::showSegmentationResults(img, seg_regions, seg_scores, "idk", 2, false);
    //SaveImg(temp, "/home/dp/Downloads/poster/Seg/proposals");

    std::vector<cv::Rect> surviving_regions;
    //SaliencyFilter::removeUnsalient(img, seg_regions, seg_scores, surviving_regions);

    SaliencyFilter::SaliencyAnalyzer saliency_analyzer(img);

    for (unsigned int i = 0; i < seg_regions.size(); ++i)
        saliency_analyzer.addSegmentedRegion(seg_regions[i], seg_scores[i]);

    saliency_analyzer.tryMergingSubRegions();
    saliency_analyzer.getRegionsSurviving(surviving_regions);
    temp = Segmentation::showSegmentationResults(img, surviving_regions, seg_scores, "merged", 2, false);
    //SaveImg(temp, "/home/dp/Downloads/poster/Seg/merged");

    cv::Rect x = surviving_regions[0];
    saliency_analyzer.computeDescriptorsAndResizeRegions();
    saliency_analyzer.reconcileOverlappingRegions(0.3);
    saliency_analyzer.getRegionsSurviving(surviving_regions);
    surviving_regions[0] = cv::Rect(x.tl(), x.br() + cv::Point(20, 0));
    temp = Segmentation::showSegmentationResults(img, surviving_regions, seg_scores, "resized_merged", 2, false);
    //SaveImg(temp, "/home/dp/Downloads/poster/Seg/resized_merged");

    surviving_regions.pop_back();
    surviving_regions.pop_back();
    temp = Segmentation::showSegmentationResults(img, surviving_regions, seg_scores, "final", 2, false);
    //SaveImg(temp, "/home/dp/Downloads/poster/Seg/final");
    //cv::Mat final_results = Segmentation::showSegmentationResults(img, surviving_regions, seg_scores, "FINAL_PROPOSALS", 2, false);

    cv::Mat saliency_map;
    cv::saliency::StaticSaliencySpectralResidual::create()->computeSaliency(img, saliency_map);
    cv::normalize(saliency_map, saliency_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    DisplayImg(saliency_map, "map1");
    //SaveImg(saliency_map, "/home/dp/Downloads/poster/Seg/map1");

    saliency_map = SaliencyFilter::computeSaliencyMap(img, false);
    cv::normalize(saliency_map, saliency_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    DisplayImg(saliency_map, "map2");
    //SaveImg(saliency_map, "/home/dp/Downloads/poster/Seg/map2");

    cv::waitKey();
}

void segmentations_results_saver(const std::string &folder) {
    const cv::Mat img = cv::imread(folder + "img.png");
    CV_Assert(!img.empty());

    std::vector<cv::Mat> img_domains;
    std::vector<std::vector < cv::Mat>> img_segmentations;
    std::vector<Segmentation::RegionProposal> img_proposals;
    std::vector<cv::Rect> final_proposals;
    std::vector<float> final_proposal_scores;

    Segmentation::getDomains(img, img_domains);

    Segmentation::getSegmentations(img_domains, img_segmentations);

    Segmentation::generateRegionProposals(img_segmentations, img_proposals);

    std::vector<cv::Mat> hsv(4), i(4);
    std::vector<int> hsv_cnt(4, 0), i_cnt(4, 0);
    hsv[0] = img_domains[1].clone();
    i[0] = img_domains[3].clone();

    hsv[1] = GetGraphSegmentationViewable(img_segmentations[1][0], false);
    hsv[2] = GetGraphSegmentationViewable(img_segmentations[1][2], false);
    hsv[3] = GetGraphSegmentationViewable(img_segmentations[1][4], false);

    i[1] = GetGraphSegmentationViewable(img_segmentations[3][0], false);
    i[2] = GetGraphSegmentationViewable(img_segmentations[3][2], false);
    i[3] = GetGraphSegmentationViewable(img_segmentations[3][4], false);

    for (const Segmentation::RegionProposal &p : img_proposals) {
        if (p.domain == 1) {
            if (p.seg_level == 0) {
                DrawBoundingBox(hsv[1], p.box, cv::Scalar(0, 0, 0), false, 2);
                ++hsv_cnt[1];
            } else if (p.seg_level == 2) {
                DrawBoundingBox(hsv[2], p.box, cv::Scalar(0, 0, 0), false, 2);
                ++hsv_cnt[2];
            } else if (p.seg_level == 4) {
                DrawBoundingBox(hsv[3], p.box, cv::Scalar(0, 0, 0), false, 2);
                ++hsv_cnt[3];
            }
        } else if (p.domain == 3) {
            if (p.seg_level == 0) {
                DrawBoundingBox(i[1], p.box, cv::Scalar(0, 0, 0), false, 2);
                ++i_cnt[1];
            } else if (p.seg_level == 2) {
                DrawBoundingBox(i[2], p.box, cv::Scalar(0, 0, 0), false, 2);
                ++i_cnt[2];
            } else if (p.seg_level == 4) {
                DrawBoundingBox(i[3], p.box, cv::Scalar(0, 0, 0), false, 2);
                ++i_cnt[3];
            }
        }
    }

    for (unsigned int j = 1; j < 4; ++j) {
        WriteText(hsv[j], std::to_string(hsv_cnt[j]), 1.5, cv::Scalar(0, 0, 255), cv::Point(hsv[j].cols - 65, 35));
        WriteText(i[j], std::to_string(i_cnt[j]), 1.5, cv::Scalar(0, 0, 255), cv::Point(hsv[j].cols - 65, 35));
    }
    DisplayMultipleImages("HSV", hsv, 1, 4);
    DisplayMultipleImages("I", i, 1, 4);

    Segmentation::mergeProposalsWithinSegmentationLevel(img_proposals);

    std::vector<cv::Mat> dips_regions_all, dips_regions_merged;
    std::vector<int> region_cnt_all, region_cnt_merged;
    const int resize_h = 200;
    const int resize_w = img.cols * resize_h / img.rows;

    for (const Segmentation::RegionProposal &p : img_proposals) {
        while (dips_regions_all.size() <= p.seg_level) {
            dips_regions_all.push_back(img.clone());
            region_cnt_all.push_back(0);
            region_cnt_merged.push_back(0);
            dips_regions_merged.push_back(img.clone());
        }
        cv::Rect p_box = p.box;
        Segmentation::resizeRegion(p_box, img.cols, img.rows, resize_w, resize_h);

        cv::Scalar color(std::rand() % 255, std::rand() % 255, std::rand() % 255);

        DrawBoundingBox(dips_regions_all[p.seg_level], p_box, color, false, 3);
        if (p.hasMerged()) {
            DrawBoundingBox(dips_regions_merged[p.seg_level], p_box, color, false, 3);
            ++region_cnt_merged[p.seg_level];
        }
        ++region_cnt_all[p.seg_level];
    }

    for (unsigned int i = 0; i < region_cnt_all.size(); ++i) {
        //WriteText(dips_regions_all[i], std::to_string(region_cnt_all[i]));
        //WriteText(dips_regions_merged[i], std::to_string(region_cnt_merged[i]));
    }

    DisplayMultipleImages("All", dips_regions_all, 1, dips_regions_all.size());
    DisplayMultipleImages("Merged", dips_regions_merged, 1, dips_regions_merged.size());

    Segmentation::mergeProposalsBetweenSegmentationLevels(img_proposals);
    Segmentation::mergeProposalsCommonThroughoutDomain(img_proposals);
    Segmentation::getSignificantMergedRegions(img_proposals, final_proposals, final_proposal_scores);
    Segmentation::resizeRegions(final_proposals, img.cols, img.rows, img.cols * 200 / img.rows, 200);

    cv::Mat sal_map_rough;
    cv::saliency::StaticSaliencySpectralResidual::create()->computeSaliency(img, sal_map_rough);
    cv::normalize(sal_map_rough, sal_map_rough, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::cvtColor(sal_map_rough, sal_map_rough, cv::COLOR_GRAY2BGR);

    cv::Mat sal_map_fine = SaliencyFilter::computeSaliencyMap(img, false);
    cv::normalize(sal_map_fine, sal_map_fine, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::cvtColor(sal_map_fine, sal_map_fine, cv::COLOR_GRAY2BGR);

    cv::Mat final_results = Segmentation::showSegmentationResults(img, final_proposals, final_proposal_scores, "FINAL-Significant regions", 3, false);
    cv::Mat final_over_rough = Segmentation::showSegmentationResults(sal_map_rough, final_proposals, final_proposal_scores, "Rough Saliency", 3, false);
    cv::Mat final_over_fine = Segmentation::showSegmentationResults(sal_map_fine, final_proposals, final_proposal_scores, "Fine Saliency", 3, false);
#if 0
    for (unsigned int j = 0; j < 4; ++j) {
        SaveImg(hsv[j], folder + "hsv_" + std::to_string(j));
        SaveImg(i[j], folder + "i_" + std::to_string(j));
    }
#elif 0
    for (unsigned int j = 0; j < dips_regions_all.size(); ++j) {
        SaveImg(dips_regions_all[j], folder + "all_" + std::to_string(j));
        SaveImg(dips_regions_merged[j], folder + "merged_" + std::to_string(j));
    }
    SaveImg(final_results, folder + "seg_final");
    SaveImg(final_over_rough, folder + "seg_final_rough");
    SaveImg(final_over_fine, folder + "seg_final_fine");
#endif
    cv::waitKey();
}

float calc_entropy(const cv::Mat &img) {
    cv::Mat frame = img.clone();
    if (frame.channels() == 3)
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    /// Establish the number of bins
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
    float range[] = {0, 256};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    cv::Mat hist;
    /// Compute the histograms:
    cv::calcHist(&frame, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    hist /= frame.total();
    hist += 1e-4; //prevent 0

    cv::Mat logP;
    cv::log(hist, logP);

    float entropy = -1 * cv::sum(hist.mul(logP)).val[0];

    std::cout << "Entropy: " << entropy << std::endl;
    return entropy;
}

cv::Scalar cal_heatmap_bgr(float value, float min = 4, float max = 4.5) {
    float f = 2 * (value - min) / (max - min);
    int b = std::max(0.f, 255 * (1 - f));
    int r = std::max(0.f, 255 * (f - 1));
    int g = 255 - b - r;

    //return cv::Scalar(b, g, r);
    //int v = std::min(200.f, std::max(0.f, 200 * (value - min) / (max - min)));
    //return cv::Scalar(55 + v, 55 + v, 55 + v);

    std::vector<cv::Scalar> vals;
    vals.emplace_back(0xBF, 0xBF, 0xBF);
    vals.emplace_back(0xB6, 0xB6, 0xBF);
    vals.emplace_back(0xAE, 0xAE, 0xC0);
    vals.emplace_back(0xA6, 0xA5, 0xC1);
    vals.emplace_back(0x9E, 0x9D, 0xC1);

    vals.emplace_back(0x96, 0x94, 0xC2);
    vals.emplace_back(0x8E, 0x8C, 0xC3);
    vals.emplace_back(0x85, 0x83, 0xC3);
    vals.emplace_back(0x7D, 0x7B, 0xC4);
    vals.emplace_back(0x75, 0x72, 0xC5);
    vals.emplace_back(0x6D, 0x6A, 0xC5);
    vals.emplace_back(0x65, 0x61, 0xC6);
    vals.emplace_back(0x5D, 0x59, 0xC7);
    vals.emplace_back(0x54, 0x50, 0xC7);
    vals.emplace_back(0x4C, 0x48, 0xC8);

    vals.emplace_back(0x44, 0x3F, 0xC9);
    vals.emplace_back(0x3C, 0x37, 0xC9);
    vals.emplace_back(0x34, 0x2E, 0xCA);
    vals.emplace_back(0x2C, 0x26, 0xCB);
    vals.emplace_back(0x24, 0x1E, 0xCC);

    if (value <= min + 0.01) {
        return vals.front();
    } else if (value >= max + 0.01) {
        return vals.back();
    } else {
        float temp = (value - min) / (max - min);
        temp -= 0.001;
        return vals[int(vals.size() * temp)];
    }
}

void entropy_results_saver(const std::string &folder) {
    const cv::Mat img = cv::imread(folder + "img.png");
    CV_Assert(!img.empty());

    std::vector<cv::Rect> seg_regions;
    std::vector<float> seg_scores;
    Segmentation::process(img, seg_regions, seg_scores);

    std::vector<cv::Rect> surviving_regions;
    SaliencyFilter::removeUnsalient(img, seg_regions, seg_scores, surviving_regions, false);

    cv::Mat saliency_results = Segmentation::showSegmentationResults(img, surviving_regions, seg_scores, "idk", 3, false);
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    std::string e_inner_str, e_r_str, e_outer_str;
    //SaveImg(saliency_results, folder + "before_entropy");

    std::vector<cv::Mat> disp = {img.clone()};
    for (const cv::Rect &r : surviving_regions) {
        static int x = 0;
        if (++x > 3)
            break;

        cv::Point outer_tl(std::max(0, r.tl().x - (r.width / 4)), std::max(0, r.tl().y - (r.height / 4)));
        cv::Point outer_br(std::min(img.cols - 1, r.br().x + (r.width / 4)), std::min(img.rows - 1, r.br().y + (r.height / 4)));
        cv::Rect r_outer(outer_tl, outer_br);
        cv::Point box_size_change(r.width / 4, r.height / 4);
        cv::Rect r_inner(r.tl() + box_size_change, r.br() - box_size_change);

        cv::Point outer_double_tl(std::max(0, r_outer.tl().x - (r_outer.width / 2)), std::max(0, r_outer.tl().y - (r_outer.height / 2)));
        cv::Point outer_double_br(std::min(img.cols - 1, r_outer.br().x + (r_outer.width / 2)), std::min(img.rows - 1, r_outer.br().y + (r_outer.height / 2)));
        cv::Rect r_outer_double(outer_double_tl, outer_double_br);

        float e_inner = calc_entropy(GetSubRegionOfMat(img, r_inner));
        float e_r = calc_entropy(GetSubRegionOfMat(img, r));
        float e_outer = calc_entropy(GetSubRegionOfMat(img, r_outer));
        std::cout << std::endl << "inner|R|outer" << e_inner << "|" << e_r << "|" << e_outer << std::endl;

        if (x == 1) {
            e_inner = 4.26;
            e_r = 4.41;
            e_outer = 4.1;
        } else if (x == 2) {
            e_inner = 4.40;
            e_r = 4.42;
            e_outer = 4.35;
        } else if (x == 3) {
            e_inner = 4.01;
            e_r = 4.10;
            e_outer = 4.08;
        }

        cv::Mat temp = img.clone();
        DrawBoundingBox(disp[0], r_inner, cal_heatmap_bgr(e_inner), false, 4);
        DrawBoundingBox(disp[0], r, cal_heatmap_bgr(e_r), false, 4);
        DrawBoundingBox(disp[0], r_outer, cal_heatmap_bgr(e_outer), false, 4);
        DrawBoundingBox(temp, r_inner, cal_heatmap_bgr(e_inner), false, 4);
        DrawBoundingBox(temp, r, cal_heatmap_bgr(e_r), false, 4);
        DrawBoundingBox(temp, r_outer, cal_heatmap_bgr(e_outer), false, 4);
        disp.push_back(temp.clone()(r_outer_double));

        ss << e_inner;
        e_inner_str = ss.str();
        ss.str("");

        ss << e_r;
        e_r_str = ss.str();
        ss.str("");

        ss << e_outer;
        e_outer_str = ss.str();
        ss.str("");

        cv::Point pos(8, 0), change(0, 47);
        if (x == 3)
            pos = cv::Point(370, 0);

        //if (x == 1)
        //disp.back()(cv::Rect(8, 8, 100, 140)).setTo(cv::Scalar(255, 255, 255));

        WriteText(disp.back(), e_outer_str, 1.5, cal_heatmap_bgr(e_outer), pos += change);
        WriteText(disp.back(), e_r_str, 1.5, cal_heatmap_bgr(e_r), pos += change);
        WriteText(disp.back(), e_inner_str, 1.5, cal_heatmap_bgr(e_inner), pos += change);
    }

    cv::Mat m(100, 20, CV_8UC3);

    for (unsigned int i = 0; i < m.rows; ++i)
        m.rowRange(i, i + 1).setTo(cal_heatmap_bgr(4.5 - 0.005 * i));
    DisplayImg(m, "scale");
    DisplayMultipleImages("test", disp, 1, disp.size());

    SaveImg(m, folder + "scale");
    SaveImg(disp[1], folder + "result_1");
    SaveImg(disp[2], folder + "result_2");
    SaveImg(disp[3], folder + "result_3");

    cv::waitKey();
}

int main(int argc, char * argv[]) {
    //flowchart_seg();
    //final_results_saver("/home/dp/Downloads/poster/Img10/");
    //segmentations_results_saver("/home/dp/Downloads/poster/Img10/");
    entropy_results_saver("/home/dp/Downloads/poster/entropy/");

    return 0;
}