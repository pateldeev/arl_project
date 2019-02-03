#ifndef YOLOINTERFACE_H
#define YOLOINTERFACE_H

#include "darknet.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

class YoloInterface {
public:
    YoloInterface(const std::string &config_file, const std::string &weights_file, const std::string &class_labels_file, float thresh = 0.5);
    ~YoloInterface(void);

    //Class is not meant to be copied or moved
    YoloInterface(const YoloInterface&) = delete;
    YoloInterface(YoloInterface&&) = delete;
    YoloInterface& operator=(const YoloInterface&) = delete;
    YoloInterface& operator=(YoloInterface&&) = delete;

    void setThresholds(float threshold = 0.5, float threshold_hier = 0.5);

    std::vector< std::pair<cv::Rect, std::pair<float, std::string>> > processImage(const cv::Mat &img);


private:
    network *m_net;
    unsigned int m_net_size;
    std::vector<std::string> m_class_names;

    float m_thresh;
    float m_thresh_hier;

    //helper functions
    cv::Mat img_yolo_to_cv(const image &img);
    image img_cv_to_yolo(const cv::Mat &img);
};

#endif