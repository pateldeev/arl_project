#include <flycapture/FlyCapture2.h>

#include "functions.h"
#include "segmentation.h"
#include "yoloInterface.h"

#include <chrono>
#include <opencv2/core.hpp>

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
    //for interfacing with camera
    FlyCapture2::Camera camera;
    FlyCapture2::Image rgbImage, rawImage;
    FlyCapture2::Error error;
    cv::Mat frame, display_frame;

    //for capturing and saving images
    cv::Mat img_captured;
    const char save_file_format[] = "/home/dp/Downloads/test/%d.png";
    const char save_file_format_video[] = "/home/dp/Downloads/test/%d.avi";
    char save_name[100];
    int save_count = 0;
    const std::vector<int> save_params = {cv::IMWRITE_PNG_COMPRESSION, 0};
    cv::VideoWriter img_video;
    const int fourcc = cv::VideoWriter::fourcc('p', 'n', 'g', ' ');

    //for performing segmentation
    std::vector<cv::Rect> segmentation_regions;
    std::vector<float> segmentation_scores;

    //for performing yolo detection
    const std::string labelFile = "/home/dp/Desktop/darknet-master/data/coco.names";
    const std::string configFile = "/home/dp/Desktop/darknet-master/cfg/yolov3.cfg";
    const std::string weightsFile = "/home/dp/Desktop/darknet-master/weights/yolov3.weights";
    YoloInterface *yolo = nullptr;

    //for timing operations
    std::chrono::high_resolution_clock::time_point t1, t2;
    double duration;

    //for control of operations
    unsigned int key = '\0';
    bool run_segmentation = false;
    bool run_yolo = false;
    bool pause_camera = false;
    bool capture_frame = false;
    bool save_captured = false;
    bool video_capture = false;
    const std::vector<std::string> help_text = {"c: capture", "w: save captured", "p: pause", "r: reset everything", "s: segment captured", "y: run yolo on captured", "v: run video capture", "Q/ESC: exit"};

    if (!startFlyCapture(camera)) { //start camera
        std::cerr << "Could not start fly capture camera! " << std::endl;
        return -1;
    }

    do {
        if (key == 's') { //run segmentation on captured image
            run_segmentation = true;
        } else if (key == 'p') { //pause camera capture
            pause_camera = true;
        } else if (key == 'r') { //resume camera capture or reset everything if not paused
            if (pause_camera) {
                pause_camera = false;
            } else {
                cv::destroyAllWindows();
                run_segmentation = run_yolo = pause_camera = capture_frame = save_captured = video_capture = false;
            }
            if (video_capture) {
                img_video.release();
                video_capture = false;
            }

        } else if (key == 'c') { //capture image
            capture_frame = true;
        } else if (key == 'w') { //save captured image
            save_captured = true;
        } else if (key == 'y') { //run yolo
            run_yolo = true;
        } else if (key == 'v' && !video_capture) {
            sprintf(save_name, save_file_format_video, save_count++);
            img_video.open(save_name, -1, 20, cv::Size(frame.cols, frame.rows), true);
            video_capture = true;
        }
        if (capture_frame) //capture frame if requested
            img_captured = frame.clone();

        if (save_captured) {
            sprintf(save_name, save_file_format, save_count++);
            cv::imwrite(save_name, frame, save_params);
            std::cout << "Saved captured frame to: " << save_name << std::endl;
            save_captured = false;
        }

        if (!pause_camera) { // Get the image
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
            cv::flip(frame, frame, -1);
        }

        if (run_segmentation) { // Run segmentation
            t1 = std::chrono::high_resolution_clock::now();
            Segmentation::process(img_captured, segmentation_regions, segmentation_scores);
            t2 = std::chrono::high_resolution_clock::now();

            duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            std::cout << std::endl << "Segmentation Time: " << duration / 1000 << "s" << std::endl;

            Segmentation::showSegmentationResults(img_captured, segmentation_regions, segmentation_scores);

            run_segmentation = false;
        }

        if (run_yolo) { //run yolo on captured image
            if (!yolo)
                yolo = new YoloInterface(configFile, weightsFile, labelFile);

            cv::imshow("yolo_detection", YoloInterface::getPredictionsDisplayable(img_captured, yolo->processImage(img_captured)));
            run_yolo = false;
        }

        display_frame = frame.clone();
        WriteText(display_frame, help_text);
        cv::imshow("camera", display_frame);

        if (capture_frame) {
            cv::imshow("img_captured", img_captured);
            capture_frame = false;
        }

        if (video_capture) {
            img_video << frame;
            static int cnt = 0;
            if (++cnt % 20 == 1) {
                sprintf(save_name, save_file_format, save_count++);
                cv::imwrite(save_name, frame, save_params);
            }
        }
        key = cv::waitKey(1);
    } while (key != 'q' && key != 27); //'q' or 'ESC' to exit


    delete yolo;
    camera.StopCapture();
    camera.Disconnect();
    std::cout << "Camera disconnected!" << std::endl;

    return 0;
}