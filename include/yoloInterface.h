#ifndef YOLOINTERFACE_H
#define YOLOINTERFACE_H

#include "darknet.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>

class YoloInterface {
public:
    YoloInterface(const std::string &config_file, const std::string &weights_file, const std::string &class_labels_file, float thresh = 0.5);
    YoloInterface(float thresh = 0.5);
    ~YoloInterface(void);

    //class is not meant to be copied. can be moved
    YoloInterface(const YoloInterface&) = delete;
    YoloInterface(YoloInterface&&) = default;
    YoloInterface& operator=(const YoloInterface&) = delete;
    YoloInterface& operator=(YoloInterface&&) = default;

    void loadNetwork(const std::string &config_file, const std::string &weights_file, const std::string &class_labels_file);

    void setThresholds(float threshold = 0.5, float threshold_hier = 0.5);

    std::vector< std::pair<cv::Rect, std::pair<float, std::string>> > processImage(const cv::Mat &img);

    void saveResults(const std::string &filename) const;
    void readResults(const std::string &filename);

    unsigned int size(void) const;
    std::pair<cv::Rect, std::pair<float, std::string>> operator[](int i);
    const std::pair<cv::Rect, std::pair<float, std::string>>& operator[](int i) const;

    static cv::Mat getPredictionsDisplayable(const cv::Mat &img, const std::vector< std::pair<cv::Rect, std::pair<float, std::string>> > &predictions);


private:
    network *m_net;
    unsigned int m_net_size;

    float m_thresh;
    float m_thresh_hier;

    std::vector<std::string> m_class_names;

    std::vector< std::pair<cv::Rect, std::pair<float, std::string>> > m_predictions;

private:
    //helper functions
    cv::Mat img_yolo_to_cv(const image &img);
    image img_cv_to_yolo(const cv::Mat &img);
    void sortPredictionsByObjectness(void);
};

#endif