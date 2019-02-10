
#include <opencv2/highgui.hpp>

#include "yoloInterface.h"

#include "functions.h"
#include "selectiveSegmentation.h"

cv::Mat getPredictionImg(YoloInterface &y, const cv::Mat &img) {
    cv::Mat disp_img = img.clone();

    std::vector< std::pair<cv::Rect, std::pair<float, std::string>> > regions = y.processImage(img);

    int text_y_loc = 1;
    for (std::pair<cv::Rect, std::pair<float, std::string>> &r : regions) {
        cv::Scalar color(100 + std::rand() % 155, 100 + std::rand() % 155, 100 + std::rand() % 155);
        std::string text = r.second.second + "(" + std::to_string(r.second.first) + ")";
        cv::rectangle(disp_img, r.first, color, 2);
        cv::putText(disp_img, text, cv::Point(10, 25 * (++text_y_loc)), cv::FONT_HERSHEY_DUPLEX, 0.75, color, 1.25);
    }

    return disp_img;
}

int main(int argc, char * argv[]) {
    const std::string labelFile = "/home/dp/Desktop/darknet-master/data/coco.names";
    const std::string configFile = "/home/dp/Desktop/darknet-master/cfg/yolov3.cfg";
    const std::string weightsFile = "/home/dp/Desktop/darknet-master/weights/yolov3.weights";

    cv::VideoCapture webCam;
    if (!webCam.open(0)) {
        std::cerr << std::endl << "Could not open webcam!" << std::endl;
        return -1;
    }

    YoloInterface yolo(configFile, weightsFile, labelFile);
    cv::Mat frame;
    bool pause = false;
    bool detect = false;


    while (true) {
        if (!pause) {
            webCam >> frame;

            if (frame.empty()) {
                std::cout << std::endl << "Video frame ended!" << std::endl;
                break;
            }

            DisplayImg((detect ? getPredictionImg(yolo, frame) : frame), "camera");
        }

        unsigned int keyPressed = cv::waitKey(1);
        if (keyPressed == 27) { //ESC
            break;
        } else if (keyPressed == 'p') {
            pause = true;
        } else if (keyPressed == 'c') {
            pause = false;
        } else if (keyPressed == 'd') {
            detect = true;
            if (pause)
                DisplayImg(getPredictionImg(yolo, frame), "camera");
        } else if (keyPressed == 'f') {
            detect = false;
            if (pause)
                DisplayImg(frame, "camera");
        } else if (keyPressed == 's' && pause) {
            std::vector<cv::Rect> proposals;
            //rescale image
            const int resizeHeight = 200;
            const int oldHeight = frame.rows, oldWidth = frame.cols;
            const int resizeWidth = oldWidth * resizeHeight / oldHeight;
            cv::Mat imgResized;
            cv::resize(frame, imgResized, cv::Size(resizeWidth, resizeHeight));
            DisplayImg(imgResized, "resized");

            Segmentation::process(imgResized, proposals, 150, 150, 0.8);

#ifdef DEBUG_SEGMENTATION
            ShowManyImages("m_images", Segmentation::m_images, 2, 3);
            std::vector<cv::Mat> temp1, temp2;
            for (unsigned int i = 0; i < Segmentation::debugDisp1.size(); ++i) {
                temp1.push_back(GetGraphSegmentationViewable(Segmentation::debugDisp1[i]));
                temp2.push_back(Segmentation::debugDisp2[i]);
            }
            unsigned int key = '-';
            int currentView = 1;
            do {
                if (key == '+' || key == '=') {
                    ++currentView;
                    currentView %= 5;
                } else if (key == '-') {
                    if (currentView > 0)
                        --currentView;
                    else
                        currentView = 4;
                }

                std::vector<cv::Mat> tempDisp1, tempDisp2;
                tempDisp1.push_back(Segmentation::m_images[currentView].clone());
                tempDisp2.push_back(Segmentation::m_images[currentView].clone());
                for (unsigned int i = currentView * 5; i < currentView * 5 + 5; ++i) {
                    tempDisp1.push_back(temp1[i].clone());
                    tempDisp2.push_back(temp2[i].clone());
                }

                ShowManyImages("m_images[]_regions", tempDisp2, 2, 3);
                ShowManyImages("m_images[]", tempDisp1, 2, 3);
                key = cv::waitKey();
            } while (key != 'q');
            cv::destroyWindow("m_images");
            cv::destroyWindow("m_images[]");
            cv::destroyWindow("m_images[]_regions");
            cv::destroyWindow("resized");
#endif
        }
    }
    return 0;
}