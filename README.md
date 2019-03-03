# ARL Research Project - Deev Patel

## Goal
Create application to identify and track unclassified objects

## Building with CMake
```bash
mkdir build
cd build
cmake ..
make
```

## Executables
* Basics from openCV's abilities
  * ./old_regions 
    * For viewing the different region proposal methods
  * ./old_saliency
    * For viewing saliency model outputs
  * ./old_contours
    * For viewing OpenCV contour seqmentation algorithm
  * ./old_filteredRegions
    * For testing out region filtration methodology
* YOLOv3
  * ./yolo_visualization
    * For visualizing the output of YOLOv3 at all 3 levels
  * ./yolo
    * C++ interface to YOLOv3
* Segementation
  * ./cv_segmentation
    * For running openCV selective segmentation
  * ./segmentation
    * For running modified segmentation algorithm
  * ./segmentation_old
    * For running old version of modified - merging down without regard to level of origin
* Video Testing
  * ./webcam
    * For running YOLOv3 & openCV selective segmentation on webcam
  * ./fly_capture_camera
    * For running YOLOv3 & modified segmentation on flycapture camera
  * ./fly_capture_camera_detailed
    * For running YOLOv3 & modified segmentation in debug mode (still displaying old domains) on flycapture camera
* ./test_area
  * For testing experimental stuff - [selective segmentation](https://www.robots.ox.ac.uk/~vgg/rg/papers/sande_iccv11.pdf) & saliency

## YOLOv3 Interface
The interface to [YOLOv3](https://github.com/pateldeev/editedYOLOv3) is linked to the darknet library. The library is created (in the build directory) from the darknet source files ("/src/yoloInterface/darknet") and the darknet makefile. So, to update the darknet library after changing its source files, rebuild the entire project. By default, darknet is built with OpenMP, Cuda, and CUDNN.

