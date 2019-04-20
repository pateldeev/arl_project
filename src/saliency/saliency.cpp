#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "saliency.h"
#include "segmentation.h"
#include "functions.h"

#include "vocus2.h"

namespace SaliencyFilter {

    void displayDebugInfo(const cv::Mat &img, const std::vector<cv::Rect> &segmentation_regions, const std::vector<float> &segmentation_scores) {
        cv::Mat img_saliency = computeSaliencyMap(img);

        std::vector<float> avg_saliency;
        for (const cv::Rect &r : segmentation_regions)
            avg_saliency.push_back(cv::mean(GetSubRegionOfMat(img_saliency, r))[0]);

        cv::Mat disp_saliency;
        cv::normalize(img_saliency, disp_saliency, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::cvtColor(disp_saliency, disp_saliency, cv::COLOR_GRAY2BGR);

        std::time_t segmentation_srand_time = std::time(0); //to have both displays show same colors
        std::srand(segmentation_srand_time);
        Segmentation::showSegmentationResults(disp_saliency, segmentation_regions, avg_saliency, "segmentations_orig_over_saliency");
        std::srand(segmentation_srand_time);
        Segmentation::showSegmentationResults(img, segmentation_regions, segmentation_scores, "segmentations_orig");
    }

    cv::Mat computeSaliencyMap(const cv::Mat &img, bool equalize) {
#if 0
        cv::Mat saliency_map;
        cv::saliency::StaticSaliencySpectralResidual::create()->computeSaliency(img, saliency_map);
        return saliency_map;
#else
        VOCUS2 vocus2;
        vocus2.process(img);
        cv::Mat sal = vocus2.get_salmap();

        if (equalize) {
            cv::Mat new_sal(sal.rows, sal.cols, CV_8UC1);
            for (unsigned int y = 0; y < sal.rows; ++y)
                for (unsigned int x = 0; x < sal.cols; ++x)
                    new_sal.at<uint8_t>(y, x) = cvRound(sal.at<float>(y, x) * 255);

            cv::equalizeHist(new_sal, new_sal);
            for (unsigned int y = 0; y < sal.rows; ++y)
                for (unsigned int x = 0; x < sal.cols; ++x)
                    sal.at<float>(y, x) = float(new_sal.at<uint8_t>(y, x)) / 255;
        }

        return sal;
#endif
    }

    void removeUnsalient(const cv::Mat &img, const std::vector<cv::Rect> &segmentation_regions, const std::vector<float> &segmentation_scores, std::vector<cv::Rect> &surviving_regions, bool display_debug) {
        if (display_debug)
            displayDebugInfo(img, segmentation_regions, segmentation_scores);

        SaliencyAnalyzer saliency_analyzer(img);
        for (unsigned int i = 0; i < segmentation_regions.size(); ++i)
            saliency_analyzer.addSegmentedRegion(segmentation_regions[i], segmentation_scores[i]);

        saliency_analyzer.tryMergingSubRegions();

#if 0
        cv::Mat sal_map = computeSaliencyMap(img);
        ShowHistrogram(sal_map);
        CreateThresholdImageWindow(sal_map);
        cv::waitKey();
        return;
#endif
        saliency_analyzer.ensureSaliencyOfRegions();

        while (saliency_analyzer.reconcileOverlappingRegions());

        saliency_analyzer.ensureDistinguishabilityOfRegions();

        if (display_debug) {
            saliency_analyzer.getRegionsSurviving(surviving_regions);
            Segmentation::showSegmentationResults(img, surviving_regions, segmentation_scores, "segmentations_to_resize");
        }
        //CreateThresholdImageWindow(computeSaliencyMap(img));
        //return;

        saliency_analyzer.computeDescriptorsAndResizeRegions();

        //saliency_analyzer.ensureRegionSaliencyFromSurroundings();

        if (display_debug) {
            saliency_analyzer.getRegionsSurviving(surviving_regions);
            Segmentation::showSegmentationResults(img, surviving_regions, segmentation_scores, "resized_segmentations");
        }

        saliency_analyzer.tryMergingSubRegions(0.3);
        saliency_analyzer.reconcileOverlappingRegions(0.3);

        if (display_debug) {
            saliency_analyzer.getRegionsSurviving(surviving_regions);
            Segmentation::showSegmentationResults(img, surviving_regions, segmentation_scores, "before_final_removal");
        }

        saliency_analyzer.keepBestRegions();

        saliency_analyzer.getRegionsSurviving(surviving_regions);
        if (display_debug)
            Segmentation::showSegmentationResults(img, surviving_regions, segmentation_scores, "FINAL_REGIONS", 2);
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