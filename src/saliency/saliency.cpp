#include <opencv2/highgui.hpp>

#include "saliency.h"
#include "segmentation.h"
#include "functions.h"

namespace SaliencyFilter {

    void displayDebugInfo(const cv::Mat &img, const std::vector<cv::Rect> &segmentation_regions, const std::vector<float> &segmentation_scores) {
        cv::Mat img_saliency;
        cv::saliency::StaticSaliencySpectralResidual::create()->computeSaliency(img, img_saliency);

        std::vector<float> avg_saliency;
        for (const cv::Rect &r : segmentation_regions)
            avg_saliency.push_back(cv::mean(GetSubRegionOfMat(img_saliency, r))[0]);

        cv::Mat disp_saliency;
        cv::normalize(img_saliency, disp_saliency, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::cvtColor(disp_saliency, disp_saliency, cv::COLOR_GRAY2BGR);

        Segmentation::showSegmentationResults(img, segmentation_regions, segmentation_scores, "segmentations");
        Segmentation::showSegmentationResults(disp_saliency, segmentation_regions, avg_saliency, "saliency_map");
    }

    void removeUnsalient(const cv::Mat &img, const std::vector<cv::Rect> &segmentation_regions, const std::vector<float> &segmentation_scores, std::vector<cv::Rect> &surviving_regions) {
        displayDebugInfo(img, segmentation_regions, segmentation_scores);
        surviving_regions.clear();

        SaliencyAnalyzer saliency_analyzer(img);
        for (int i = 0; i < segmentation_regions.size(); ++i)
            saliency_analyzer.addSegmentedRegion(segmentation_regions[i], segmentation_scores[i]);

        saliency_analyzer.m_img = img.clone();

        saliency_analyzer.mergeSubRegions();

        saliency_analyzer.applySaliencyThreshold();

        saliency_analyzer.getRegionsSurviving(surviving_regions);
        cv::Mat disp = img.clone();
        for (const cv::Rect &r : surviving_regions)
            DrawBoundingBox(disp, r, cv::Scalar(255, 0, 0));
        DisplayImg(disp, "Surviving_Regions_");
        surviving_regions.clear();

        saliency_analyzer.computeSaliencyDescriptors();

        saliency_analyzer.getRegionsSurviving(surviving_regions);
        disp = img.clone();
        for (const cv::Rect &r : surviving_regions)
            DrawBoundingBox(disp, r, cv::Scalar(255, 0, 0));
        DisplayImg(disp, "Surviving_Regions");
    }

    void calculateMeanSTD(const std::vector<float> &data, float &mean, float &stdeviation) {
        float sum = 0.0;
        stdeviation = 0.0;
        for (float d : data)
            sum += d;
        mean = sum / data.size();

        for (float d : data)
            stdeviation += std::pow(d - mean, 2);

        stdeviation = std::sqrt(stdeviation / data.size());
    }
};