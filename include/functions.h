#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/ximgproc/segmentation.hpp>

#include <opencv2/saliency/saliencyBaseClasses.hpp>
#include <opencv2/saliency/saliencySpecializedClasses.hpp>

void CalculateRegionProposals(const cv::Mat & img, std::vector<cv::Rect> & regions, const int resizeHeight = 200);

void CalculateSaliceny(const cv::Mat & img, cv::Mat & saliencyMat, bool useFineGrained = true);

void RemoveUnsalient(const cv::Mat & salientImg, const std::vector<cv::Rect> & regions, std::vector<cv::Rect> & highest, const int keepNum = 7);

#endif