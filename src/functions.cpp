#include "functions.h"


#include <numeric>

cv::Mat GetSubRegionOfMat(const cv::Mat &m, const cv::Rect &r) {
    return m(cv::Range(r.tl().y, r.br().y + 1), cv::Range(r.tl().x, r.br().x + 1));
}

void SaveImg(const cv::Mat &img, const std::string &loc) {
    static const std::vector<int> save_params = {cv::IMWRITE_PNG_COMPRESSION, 0}; //parameters needed to save in lossless png mode
    //save -- add .png extension if not given
    cv::imwrite(((loc.find_last_of(".png") == loc.size() - 1) ? loc : loc + ".png"), img, save_params);
}

char DisplayImg(const cv::Mat &img, const std::string &window_name, int width, int height, bool wait) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, width, height);
    cv::imshow(window_name, img);

    return (wait) ? cv::waitKey() : 0;
}

char DisplayImg(const std::string &window_name, const cv::Mat &img, int width, int height, bool wait) {
    return DisplayImg(img, window_name, width, height, wait);
}

//show multiple CV_8UC1 or CV_8UC3 images in same window - doesn't check number of images provided to make sure it fits

cv::Mat DisplayMultipleImages(const std::string &window_name, const std::vector<cv::Mat> &images, unsigned int rows, unsigned int cols, const cv::Size &display_size, bool wait) {
    cv::Mat img_disp = cv::Mat::zeros(display_size.height, display_size.width, CV_8UC3);
    const int img_target_width = display_size.width / cols, img_target_height = display_size.height / rows; //target size of each small subimage

    int w = 0, h = 0; //used to keep track of top left corner of each subimage
    cv::Mat img_resized;

    for (const cv::Mat &img : images) {
        CV_Assert(img.type() == CV_8UC1 || img.type() == CV_8UC3);

        //consider the two possible resizing options - must choose optimal one that takes up most space
        int img_resize_byH_width = img.cols * (float(img_target_height) / img.rows);
        int img_resize_byW_height = img.rows * (float(img_target_width) / img.cols);

        if (img_resize_byH_width <= img_target_width) //rescale so height fits perfectly
            cv::resize(img, img_resized, cv::Size(img_resize_byH_width, img_target_height));
        else //rescale so width fits perfectly
            cv::resize(img, img_resized, cv::Size(img_target_width, img_resize_byW_height));

        //copy over relevant pixel values
        for (unsigned int r = 0; r < img_resized.rows; ++r)
            for (unsigned int c = 0; c < img_resized.cols; ++c)
                img_disp.at<cv::Vec3b>(r + h, c + w) = ((img.type() == CV_8UC3) ? img_resized.at<cv::Vec3b>(r, c) : cv::Vec3b(img_resized.at<uint8_t>(r, c), img_resized.at<uint8_t>(r, c), img_resized.at<uint8_t>(r, c)));

        //calculate start position of next image
        w += img_target_width;
        if (w >= display_size.width - img_target_width + 1) { //go to next row if needed

            w = 0;
            h += img_target_height;
        }
    }
    DisplayImg(img_disp, window_name, display_size.width, display_size.height, wait);
    return img_disp;
}

char UpdateImg(const cv::Mat &img, const std::string &window_name, const std::string &window_title, bool wait) {
    cv::imshow(window_name, img);
    if (*window_title.c_str())
        cv::setWindowTitle(window_name, window_title);
    if (wait)
        cv::waitKey();
}

void CreateThresholdImageWindow(const cv::Mat &img, const std::string &window_name_main) {
    auto trackbarCallback = [](int thresh, void* data)->void {
        cv::Mat img = ((cv::Mat*) data)->clone();
        cv::threshold(img, img, thresh, 255, cv::THRESH_BINARY);
        cv::imshow("Thresholded", img);
    };

    int threshold_val;
    cv::Mat disp_img;
    cv::normalize(img, disp_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::namedWindow(window_name_main, cv::WINDOW_NORMAL);
    cv::namedWindow("Thresholded", cv::WINDOW_NORMAL);

    cv::imshow(window_name_main, disp_img);
    cv::imshow("Thresholded", disp_img);
    cv::createTrackbar("Thresh", window_name_main, &threshold_val, 255, trackbarCallback, &disp_img);
    cv::waitKey();
    cv::destroyWindow(window_name_main);
    cv::destroyWindow("Thresholded");
}

void ShowHistrogram(const cv::Mat &img, const std::string &window_name) {
    CV_Assert(img.type() == CV_8UC1 || img.type() == CV_32F);
    if (img.type() == CV_32F) {
        double min, max;
        cv::minMaxLoc(img, &min, &max);
        CV_Assert(min >= -0.01 && max <= 1.01);
    }

    unsigned int histogram[256]; // allocate memory for no of pixels for each intensity value
    for (unsigned int i = 0; i < 255; ++i)
        histogram[i] = 0;

    // calculate the no of pixels for each intensity values
    for (unsigned int y = 0; y < img.rows; ++y)
        for (unsigned int x = 0; x < img.cols; ++x)
            if (img.type() == CV_8UC1)
                ++histogram[u_int(img.at<uchar>(y, x))];
            else
                ++histogram[u_int(255 * img.at<float>(y, x))];

    // draw the histograms
    const cv::Size hist_size(512, 400);
    const int bin_w = cvRound(double(hist_size.width) / 256);
    cv::Mat hist_img(hist_size.height, hist_size.width, CV_8UC1, cv::Scalar(255, 255, 255));

    // find the maximum intensity element from histogram
    unsigned int max = histogram[0];
    for (unsigned int i = 1; i < 256; ++i) {
        if (max < histogram[i])
            max = histogram[i];
    }

    for (unsigned int i = 0; i < 255; ++i) // normalize the histogram between 0 and hist_img.rows
        histogram[i] = u_int((double(histogram[i]) / max) * hist_img.rows);

    for (unsigned int i = 0; i < 255; ++i) // draw the intensity line for histogram
        cv::line(hist_img, cv::Point(bin_w * i, hist_size.height), cv::Point(bin_w * i, hist_size.height - histogram[i]), cv::Scalar(0, 0, 0), 1, 8, 0);

    DisplayImg(hist_img, window_name); // display histogram
}

//draw single bounding box

void DrawBoundingBox(cv::Mat &img, const cv::Rect &rect, const cv::Scalar &color, bool show_center, unsigned int thickness) {
    cv::rectangle(img, rect.tl(), rect.br(), color, thickness);
    if (show_center)
        cv::drawMarker(img, (rect.tl() + rect.br()) / 2, color);
}


//draw multiple bounding boxes

void DrawBoundingBoxes(cv::Mat &img, const std::vector<cv::Rect> &regions, const cv::Scalar &color) {
    for (const cv::Rect &rect : regions)
        cv::rectangle(img, rect.tl(), rect.br(), color);
}

void DisplayBoundingBoxesInteractive(const cv::Mat &img, const std::vector<cv::Rect> &regions, const std::string &window_name, const unsigned int increment) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1200, 1200);
    unsigned int numRectsShown = increment; // number of region proposals to show

    while (1) {
        cv::Mat dispImg = img.clone();

        //display only required number of proposals
        for (unsigned int i = 0; i < regions.size(); ++i) {
            if (i < numRectsShown)
                cv::rectangle(dispImg, regions[i], cv::Scalar(0, 255, 0));
            else
                break;
        }

        DisplayImg(dispImg, window_name);

        int k = cv::waitKey(); // record key press
        if (k == 'm')
            numRectsShown += increment; // increase total number of rectangles to show by increment
        else if (k == 'l' && numRectsShown > increment)
            numRectsShown -= increment; // decrease total number of rectangles to show by increment
        else if (k == 'c')
            return;
    }
}

void WriteText(cv::Mat &img, const std::string &text, float font_size, const cv::Scalar &font_color, const cv::Point &pos) {
    cv::putText(img, text, pos, cv::FONT_HERSHEY_DUPLEX, font_size, font_color, 1.25);
}

void WriteText(cv::Mat &img, const std::vector<std::string> &text, float font_size, const cv::Scalar &font_color, const cv::Point &start_pos, const cv::Point &change_per_line) {
    cv::Point pos = start_pos;
    for (const std::string &t : text) {
        WriteText(img, t, font_size, font_color, pos);
        pos += change_per_line;
    }
}

//returns viewable representation of graph segmentation

cv::Mat GetGraphSegmentationViewable(const cv::Mat &img_segmented, bool disp_count) {
    CV_Assert(img_segmented.type() == CV_32S);

    double min, max;
    cv::minMaxLoc(img_segmented, &min, &max);

    cv::Mat img_disp = cv::Mat::zeros(img_segmented.rows, img_segmented.cols, CV_8UC3);

    const uint32_t* p;
    uint8_t* p2;

    //helper function to determine color
    auto color_mapping = [](int segment_id) -> cv::Scalar {
        double base = double(segment_id) * 0.618033988749895 + 0.24443434;
        cv::Scalar c(std::fmod(base, 1.2), 0.95, 0.80);
        cv::Mat in(1, 1, CV_32FC3), out(1, 1, CV_32FC3);

        float * p = in.ptr<float>(0);
        p[0] = float(c[0]) * 360.0f;
        p[1] = float(c[1]);
        p[2] = float(c[2]);

        cv::cvtColor(in, out, cv::COLOR_HSV2RGB);
        cv::Scalar t;
        cv::Vec3f p2 = out.at<cv::Vec3f>(0, 0);

        t[0] = int(p2[0] * 255);
        t[1] = int(p2[1] * 255);
        t[2] = int(p2[2] * 255);

        return t;
    };

    for (int i = 0; i < img_segmented.rows; ++i) {
        p = img_segmented.ptr<uint32_t>(i);
        p2 = img_disp.ptr<uint8_t>(i);

        for (int j = 0; j < img_segmented.cols; ++j) {
            cv::Scalar color = color_mapping(int(p[j]));
            p2[j * 3] = (uint8_t) color[0];
            p2[j * 3 + 1] = (uint8_t) color[1];
            p2[j * 3 + 2] = (uint8_t) color[2];
        }
    }

    if (disp_count)
        WriteText(img_disp, std::to_string(int(max + 1)));

    return img_disp;
}

//make sure grid lines fit evenly for optimal behavior

void DrawGridLines(cv::Mat &img, int num_grids, const cv::Scalar &grid_color) {
    const int grid_size_r = img.rows / num_grids;
    const int grid_size_c = img.cols / num_grids;

    for (int r = 0; r < img.rows; r += grid_size_r)
        cv::line(img, cv::Point(r, 0), cv::Point(r, img.cols), grid_color);

    for (int c = 0; c < img.cols; c += grid_size_c)
        cv::line(img, cv::Point(0, c), cv::Point(img.rows, c), grid_color);
}

float CalcSpatialEntropy(const cv::Mat &img, const cv::Rect &region, const cv::Rect &region_blackout) {
    CV_Assert((region_blackout & region) == region_blackout);
    CV_Assert(region.height < img.rows && region.width < img.cols);
    cv::Mat img_sub = img(region).clone();
    img_sub(region_blackout - region.tl()).setTo(0);

    return CalcSpatialEntropy(img_sub);
}

float CalcSpatialEntropy(const cv::Mat &img, const cv::Rect &region) {
    CV_Assert(region.height < img.rows && region.width < img.cols);
    return CalcSpatialEntropy(img(region));
}

float CalcSpatialEntropy(const cv::Mat &img) {
    //Compute Sobel Derivate of the WHOLE Image as it needs to be globally normalized and threshold but regional entropy is to be calculated
    cv::Mat blur_img, gradientX, gradientY, gradient_magitude;
    cv::GaussianBlur(img, blur_img, cv::Size(3, 3), 0, 0);
    cv::Scharr(blur_img, gradientX, CV_32FC1, 1, 0);
    cv::Scharr(blur_img, gradientY, CV_32FC1, 0, 1);
    cv::magnitude(gradientX, gradientY, gradient_magitude);

    //Normalize and Threshold
    cv::Mat gradient_scaled, histogram_input;
    cv::normalize(gradient_magitude, gradient_scaled, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cv::threshold(gradient_scaled, histogram_input, 0, 255, cv::THRESH_TOZERO | cv::THRESH_OTSU);

    //double min, max;
    //cv::minMaxLoc(gradient_magitude, &min, &max);

    //Histogram Parameters
    cv::Mat histogram;
    const static int kHist_bins_ = 256; //TODO: Make param
    const int num_bins = (kHist_bins_ % 2 != 0) ? kHist_bins_ : kHist_bins_ + 1; //Use odd number of bins for evenly weighted bins //Don't use a binary histogram a as bin 1 never gets populated
    float range[] = {0, 255};
    const float *histRange = {range};

    //Create Gaussian Weight Vector
    const static cv::Mat kWeights_ = cv::getGaussianKernel(num_bins, -1, CV_32FC1); //num_bins should be odd
    //std::cout << "kWeights_ = " << kWeights_.t() << std::endl;

    //Get Gradient Image Patch
    cv::Mat gradient_patch = img;

    //Compute Histogram of Image Patch
    cv::calcHist(&gradient_patch, 1, 0, cv::noArray(), histogram, 1, &num_bins, &histRange, true, false);

    //Compute Probability by dividing by total number of counts
    cv::Mat probabilty = histogram / (img.rows * img.cols); // 1 x num_bins vector

    //Compute log of probability
    cv::Mat log_probabilty;
    cv::log(probabilty, log_probabilty); //0 becomes infinity

    //Compute Absolute
    cv::Mat log_absolute = cv::abs(log_probabilty); //cv::abs -> output matrix has the same size and the same TYPE

    //Filter Inf values
    cv::Mat log_mask = cv::Mat(log_absolute != std::numeric_limits<float>::infinity());
    cv::Mat log_filtered = cv::Mat::zeros(log_probabilty.size(), log_probabilty.type());
    log_probabilty.copyTo(log_filtered, log_mask);

    //Weighted Probability
    probabilty = probabilty.mul(kWeights_);

    //Compute Entropy
    return std::abs(-1.0 * probabilty.dot(log_filtered)); //-p*log(p) //abs to make -0 to 0
}

cv::Scalar GetRandomColor(bool reset_seed) {
    static unsigned int seed = 0;
    if (reset_seed) {
        seed = 0;
        std::srand(std::time(NULL));
    } else {
        ++seed;
    }

    if (seed == 0)
        return cv::Scalar(0, 0, 0);
    else if (seed == 1)
        return cv::Scalar(0, 255, 0);
    else if (seed == 2)
        return cv::Scalar(255, 200, 0); //cv::Scalar(255, 0, 0);
    else if (seed == 3)
        return cv::Scalar(0, 0, 255);
    else if (seed == 4)
        return cv::Scalar(255, 0, 200);
    else
        return cv::Scalar(55 + std::rand() % 200, 55 + std::rand() % 200, 55 + std::rand() % 200);
}