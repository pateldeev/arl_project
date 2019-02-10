#include <flycapture/FlyCapture2.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui.hpp>

#include "functions.h"

#include "selectiveSegmentation.h"

#include <chrono>
#include <thread>
#include <opencv2/imgcodecs.hpp>

bool startFlyCapture(FlyCapture2::Camera &camera) {
    FlyCapture2::Error error;
    FlyCapture2::CameraInfo camInfo;

    // Connect the camera
    error = camera.Connect(0);
    if (error != FlyCapture2::PGRERROR_OK) {
        std::cerr << "Failed to connect to camera" << std::endl;
        return false;
    }

    // Get the camera info and print it out
    error = camera.GetCameraInfo(&camInfo);
    if (error != FlyCapture2::PGRERROR_OK) {
        std::cerr << "Failed to get camera info from camera" << std::endl;
        return false;
    }
    std::cout << "Connected to: " << camInfo.vendorName << ' ' << camInfo.modelName << ' ' << camInfo.serialNumber << std::endl;

    error = camera.StartCapture();
    if (error == FlyCapture2::PGRERROR_ISOCH_BANDWIDTH_EXCEEDED) {
        std::cerr << "Bandwidth exceeded" << std::endl;
        return false;
    } else if (error != FlyCapture2::PGRERROR_OK) {
        std::cerr << "Failed to start image capture" << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char * argv[]) {
    FlyCapture2::Camera camera;

    if (!startFlyCapture(camera)) {
        std::cerr << "Could not start fly capture camera! " << std::endl;
        return -1;
    }

    FlyCapture2::Image rgbImage, rawImage;
    FlyCapture2::Error error;

    unsigned int key = '\0';
    bool segment = false;
    bool pause = false;
    bool capture = false;
    int disp_domain = 0;

    cv::Mat frame;
    const std::vector<float> k_values = {500, 600, 700, 800, 900};

    cv::Mat img;
    std::vector<cv::Mat> img_domains;
    std::vector< std::vector<cv::Mat> > img_segmentations;

    std::chrono::high_resolution_clock::time_point t1, t2;
    double duration;

    char save_file_format[] = "/home/dp/Downloads/data/%d.png";
    char save_file[100];
    int save_count = 0;
    const std::vector<int> write_params = {cv::IMWRITE_PNG_COMPRESSION, 0};

    do {
        if (key == 's')
            segment = pause = true;
        else if (key == 'd')
            segment = pause = false;
        else if (key == '+' || key == '=')
            ++disp_domain %= img_domains.size();
        else if (key == '-' && --disp_domain < 0)
            disp_domain += img_domains.size();
        else if (key == 'g')
            capture = true;

        if (capture) {
            sprintf(save_file, save_file_format, save_count++);
            cv::imwrite(save_file, frame, write_params);
            std::cout << "Captured to: " << save_file << std::endl;
            capture = false;
        }

        if (!pause) { // Get the image
            error = camera.RetrieveBuffer(&rawImage);
            if (error != FlyCapture2::PGRERROR_OK) {
                std::cout << "capture error" << std::endl;
                continue;
            }

            // convert to rgb
            rawImage.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage);

            // convert to OpenCV Mat
            unsigned int rowBytes = double(rgbImage.GetReceivedDataSize()) / double(rgbImage.GetRows());
            frame = cv::Mat(rgbImage.GetRows(), rgbImage.GetCols(), CV_8UC3, rgbImage.GetData(), rowBytes);

        } else if (segment) { // Run segmentation
            img = frame.clone();

            t1 = std::chrono::high_resolution_clock::now();
            Segmentation::getDomains(img, img_domains);
            Segmentation::getSegmentations(img_domains, img_segmentations, k_values);
            t2 = std::chrono::high_resolution_clock::now();

            duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            std::cout << std::endl << "Segmentation Time: " << duration / 1000 << "s" << std::endl;

            segment = false;
        } else {
            DisplayImg(img, "image_segmentation");
            ShowManyImages("domains", img_domains, 2, 3);

            std::vector<cv::Mat> temp;
            temp.push_back(img_domains[disp_domain]);
            for (const cv::Mat &s : img_segmentations[disp_domain])
                temp.push_back(GetGraphSegmentationViewable(s));

            ShowManyImages("segmentations", temp, 2, 3);
        }

        cv::imshow("camera", frame);

        key = cv::waitKey(1);
    } while (key != 'q' && key != 'c');

    camera.StopCapture();
    camera.Disconnect();
    std::cout << "Camera disconnected!" << std::endl;

    return 0;
}