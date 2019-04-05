#include "yoloInterface.h"

#include <iostream>

YoloInterface::YoloInterface(const std::string &config_file, const std::string &weights_file, const std::string &class_labels_file, float thresh)
: m_net_size(0), m_thresh(thresh), m_thresh_hier(0.5) {
    std::cout << std::endl << "Loading darknet network!" << std::endl;

    //load network 
    m_net = load_network(const_cast<char*> (config_file.c_str()), const_cast<char*> (weights_file.c_str()), 0);
    m_net_size = m_net->n;
    set_batch_network(m_net, 1);

    //get class names and convert to more usable format
    char **names = get_labels(const_cast<char*> (class_labels_file.c_str()));
    m_class_names.resize(m_net->layers[m_net_size - 1].classes);
    for (int i = 0; i < m_class_names.size(); ++i)
        m_class_names[i] = names[i];
    free((void *) names);

    std::cout << std::endl << "Done Loading darknet network!" << std::endl;
}

YoloInterface::~YoloInterface(void) {
    free_network(m_net);
}

void YoloInterface::setThresholds(float threshold, float threshold_hier) {
    m_thresh = threshold;
    m_thresh_hier = threshold_hier;
}

std::vector< std::pair<cv::Rect, std::pair<float, std::string>> > YoloInterface::processImage(const cv::Mat &img) {
    image img_yolo = img_cv_to_yolo(img);
    image img_resized = letterbox_image(img_yolo, m_net->w, m_net->h);
    layer l = m_net->layers[m_net_size - 1];

    double time = what_time_is_it_now();
    network_predict(m_net, img_resized.data);
    //printf("\nPredicted in %f seconds.\n", (what_time_is_it_now() - time));

    int nboxes = 0;
    detection *dets = get_network_boxes(m_net, img_yolo.w, img_yolo.h, m_thresh, m_thresh_hier, 0, 1, &nboxes);

    do_nms_sort(dets, nboxes, l.classes, 0.45);

    std::vector< std::pair<cv::Rect, std::pair<float, std::string>> > regions;
    regions.reserve(nboxes);

    for (int i = 0; i < nboxes; ++i) {
        cv::Point r_tl(img_yolo.w * (dets[i].bbox.x - dets[i].bbox.w / 2), img_yolo.h * (dets[i].bbox.y - dets[i].bbox.h / 2));
        cv::Size r_scale(dets[i].bbox.w * img_yolo.w, dets[i].bbox.h * img_yolo.h);

        for (int j = 0; j < m_class_names.size(); ++j) {
            if (dets[i].prob[j] > m_thresh)
                regions.emplace_back(std::piecewise_construct, std::forward_as_tuple(r_tl, r_scale), std::forward_as_tuple(dets[i].prob[j], m_class_names[j]));
        }
    }

    free_detections(dets, nboxes);
    free_image(img_resized);
    free_image(img_yolo);

    sortPredictionsByObjectness(regions);

    return regions;
}

void YoloInterface::sortPredictionsByObjectness(std::vector< std::pair<cv::Rect, std::pair<float, std::string>> > &predictions) {
    std::sort(predictions.begin(), predictions.end(), [](const std::pair<cv::Rect, std::pair<float, std::string>> &p1, const std::pair<cv::Rect, std::pair<float, std::string>> &p2)->bool {
        return p1.second.first > p2.second.first;
    });
}

cv::Mat YoloInterface::getPredictionsDisplayable(const cv::Mat &img, const std::vector< std::pair<cv::Rect, std::pair<float, std::string>> > &predictions) {
    cv::Mat disp_img = img.clone();

    int text_y_loc = 1;
    for (const std::pair<cv::Rect, std::pair<float, std::string>> &r : predictions) {
        cv::Scalar color(100 + std::rand() % 155, 100 + std::rand() % 155, 100 + std::rand() % 155);
        std::string text = r.second.second + "(" + std::to_string(r.second.first) + ")";
        cv::rectangle(disp_img, r.first, color, 2);
        cv::putText(disp_img, text, cv::Point(10, 25 * (++text_y_loc)), cv::FONT_HERSHEY_DUPLEX, 0.75, color, 1.25);
    }

    return disp_img;
}

cv::Mat YoloInterface::img_yolo_to_cv(const image &img) {
    cv::Mat img_mat = cv::Mat(img.h, img.w, CV_8UC3);
    const float *img_data = img.data;
    for (int ch = 0; ch < img.c; ++ch)
        for (int row = 0; row < img.h; ++row)
            for (int col = 0; col < img.w; ++col)
                img_mat.at<cv::Vec3b>(row, col)[2 - ch] = uint8_t(255 * img_data++[0]); //OpenCV stores in BGR format

    return img_mat;
}

image YoloInterface::img_cv_to_yolo(const cv::Mat &img) {
    CV_Assert(img.type() == CV_8UC3);
    image img_yolo = make_image(img.cols, img.rows, 3);

    float *img_data = img_yolo.data;
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < img.rows; ++j)
            for (int i = 0; i < img.cols; ++i)
                img_data++[0] = float(img.at<cv::Vec3b>(j, i)[2 - k]) / 255; //OpenCV stores in BGR format

    return img_yolo;
}