#include <flycapture/FlyCapture2.h>

#include "functions.h"
#include "segmentation.h"
#include "yoloInterface.h"

#include <chrono>

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
    const char save_file_format[] = "/home/dp/Downloads/data/%d.png";
    char save_file_name[100];
    int save_count = 0;
    const std::vector<int> save_params = {cv::IMWRITE_PNG_COMPRESSION, 0};

    //for performing segmentation
    const std::vector<float> k_values = {500, 600, 700, 800, 900};
    int disp_domain = 0;
    std::vector<cv::Mat> img_domains;
    std::vector< std::vector<cv::Mat> > img_segmentations;

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
    bool update_segmentation_display = false;
    bool run_yolo = false;
    bool pause_camera = false;
    bool capture_frame = false;
    bool save_captured = false;
    const std::vector<std::string> help_text = {"c: capture", "w: save captured", "p: pause", "r: reset everything", "s: segment captured", "   +/-: change segmentation display domain", "y: run yolo on captured", "Q/ESC: exit"};

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
                run_segmentation = update_segmentation_display = run_yolo = pause_camera = capture_frame = save_captured = false;
            }
        } else if (key == '+' || key == '=') { //view next domain for segmentation
            ++disp_domain %= img_domains.size();
            update_segmentation_display = true;
        } else if (key == '-') { //view previous domain for segmentation
            if (--disp_domain < 0)
                disp_domain += img_domains.size();
            update_segmentation_display = true;
        } else if (key == 'c') { //capture image
            capture_frame = true;
        } else if (key == 'w') { //save captured image
            save_captured = true;
        } else if (key == 'y') { //run yolo
            run_yolo = true;
        }

        if (capture_frame) //capture frame if requested
            img_captured = frame.clone();

        if (save_captured) {
            sprintf(save_file_name, save_file_format, save_count++);
            cv::imwrite(save_file_name, frame, save_params);
            std::cout << "Saved captured frame to: " << save_file_name << std::endl;
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
        }

        if (run_segmentation) { // Run segmentation
            t1 = std::chrono::high_resolution_clock::now();
            Segmentation::getDomains(img_captured, img_domains);
            Segmentation::getSegmentations(img_domains, img_segmentations, k_values);
            t2 = std::chrono::high_resolution_clock::now();

            duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            std::cout << std::endl << "Segmentation Time: " << duration / 1000 << "s" << std::endl;

            run_segmentation = false;
            update_segmentation_display = true;
        }

        if (update_segmentation_display) { //display segmentation results
            std::vector<cv::Mat> temp;
            temp.push_back(img_domains[disp_domain].clone());
            for (const cv::Mat &s : img_segmentations[disp_domain])
                temp.push_back(GetGraphSegmentationViewable(s));

            WriteText(temp[0], std::to_string(disp_domain));
            ShowManyImages("domains_all", img_domains, 2, 3);
            ShowManyImages("selected_domain_segmentations", temp, 2, 3);

            update_segmentation_display = false;
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

        key = cv::waitKey(1);
    } while (key != 'q' && key != 27); //'q' or 'ESC' to exit


    delete yolo;
    camera.StopCapture();
    camera.Disconnect();
    std::cout << "Camera disconnected!" << std::endl;

    return 0;
}