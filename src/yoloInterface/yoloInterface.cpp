#include "yoloInterface.h"

#include "functions.h"

YoloInterface::YoloInterface(const std::string &config_file, const std::string &weights_file, const std::string &class_labels_file, float thresh)
: YoloInterface(thresh) {
    loadNetwork(config_file, weights_file, class_labels_file); //load network 
}

YoloInterface::YoloInterface(float thresh) : m_net(nullptr), m_net_size(0), m_thresh(thresh), m_thresh_hier(0.5) {
}

YoloInterface::~YoloInterface(void) {
    free_network(m_net);
}

void YoloInterface::loadNetwork(const std::string &config_file, const std::string &weights_file, const std::string &class_labels_file) {
    m_net = load_network(const_cast<char*> (config_file.c_str()), const_cast<char*> (weights_file.c_str()), 0);
    m_net_size = m_net->n;
    set_batch_network(m_net, 1);

    //get class names and convert to more usable format
    char **names = get_labels(const_cast<char*> (class_labels_file.c_str()));
    m_class_names.resize(m_net->layers[m_net_size - 1].classes);
    for (unsigned int i = 0; i < m_class_names.size(); ++i)
        m_class_names[i] = names[i];
    free((void *) names);
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


    m_predictions.clear();
    m_predictions.reserve(nboxes);

    for (int i = 0; i < nboxes; ++i) {
        cv::Point r_tl(img_yolo.w * (dets[i].bbox.x - dets[i].bbox.w / 2), img_yolo.h * (dets[i].bbox.y - dets[i].bbox.h / 2));
        cv::Size r_scale(dets[i].bbox.w * img_yolo.w, dets[i].bbox.h * img_yolo.h);

        for (unsigned int j = 0; j < m_class_names.size(); ++j) {
            if (dets[i].prob[j] > m_thresh)
                m_predictions.emplace_back(std::piecewise_construct, std::forward_as_tuple(r_tl, r_scale), std::forward_as_tuple(dets[i].prob[j], m_class_names[j]));
        }
    }

    free_detections(dets, nboxes);
    free_image(img_resized);
    free_image(img_yolo);

    sortPredictionsByObjectness();

    return m_predictions;
}

void YoloInterface::saveResults(const std::string &filename) const {
    std::ofstream f(filename);
    if (!f.is_open())
        return;

    for (const std::pair<cv::Rect, std::pair<float, std::string>> &p : m_predictions)
        f << "(" << p.first.tl().x << "," << p.first.tl().y << ")-->" << "(" << p.first.br().x << "," << p.first.br().y << ")" << p.second.second << "=" << p.second.first << std::endl;

    f.close();
}

void YoloInterface::readResults(const std::string &filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        return;

    m_predictions.clear();


    cv::Point tl, br;
    std::string class_name;
    float objectness;

    std::string s;
    unsigned int p1, p2;

    while (std::getline(f, s) && s.size()) {
        p1 = 0;
        p2 = s.find_first_of(',');
        tl.x = std::stoi(s.substr(p1 + 1, p2 - p1 - 1));

        p1 = p2;
        p2 = s.find_first_of(')', p1);
        tl.y = std::stoi(s.substr(p1 + 1, p2 - p1 - 1));

        p1 = s.find_first_of('(', p2);
        p2 = s.find_first_of(',', p1);
        br.x = std::stoi(s.substr(p1 + 1, p2 - p1 - 1));

        p1 = p2;
        p2 = s.find_first_of(')', p1);
        br.y = std::stoi(s.substr(p1 + 1, p2 - p1 - 1));

        p1 = p2;
        p2 = s.find_first_of('=', p1);
        class_name = s.substr(p1 + 1, p2 - p1 - 1);

        p1 = p2;
        objectness = std::stof(s.substr(p1 + 1));

        m_predictions.emplace_back(std::piecewise_construct, std::forward_as_tuple(tl, br), std::forward_as_tuple(objectness, class_name));
    }

    f.close();
}

unsigned int YoloInterface::size(void) const {
    return m_predictions.size();
}

std::pair<cv::Rect, std::pair<float, std::string>> YoloInterface::operator[](int i) {
    //std::cout << "2" << std::endl;
    return m_predictions.at(i);
}

const std::pair<cv::Rect, std::pair<float, std::string>>&YoloInterface::operator[](int i) const {
    //std::cout << "1" << std::endl;
    return m_predictions.at(i);
}

cv::Mat YoloInterface::getPredictionsDisplayable(const cv::Mat &img, const std::vector< std::pair<cv::Rect, std::pair<float, std::string>> > &predictions) {
    cv::Mat disp_img = img.clone();

    int text_y_loc = 1;
    //std::srand(std::time(NULL));
    GetRandomColor(true);
    for (const std::pair<cv::Rect, std::pair<float, std::string>> &r : predictions) {
        static int i = 0;
        if (++i == 1)
            continue;
        cv::Scalar color = GetRandomColor();
        std::string text = r.second.second + "(" + std::to_string(r.second.first) + ")";
        cv::rectangle(disp_img, r.first, color, 2);
        cv::putText(disp_img, text, cv::Point(10, 42 * (++text_y_loc)), cv::FONT_HERSHEY_DUPLEX, 1.2, color, 1.25);
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

void YoloInterface::sortPredictionsByObjectness(void) {
    std::sort(m_predictions.begin(), m_predictions.end(), [](const std::pair<cv::Rect, std::pair<float, std::string>> &p1, const std::pair<cv::Rect, std::pair<float, std::string>> &p2)->bool {
        return p1.second.first > p2.second.first;
    });
}
