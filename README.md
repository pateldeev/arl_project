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
* ./regions 
  * For viewing the different region proposal methods
* ./saliency
  * For view saliency methods
* ./contours
  * For viewing OpenCV contour seqmentation algorithm
* ./filteredRegions
  * For testing out region filtration methodology
* ./testarea
  * For testing experimental stuff - selective segmentation
* ./yoloVisualization
  * For visualizing the output of YOLO at all 3 levels
* ./yolo
  * Interface to YOLOv3

## YOLOv3 Interface
The interface to YOLOv3 is linked to the darknet library. The library is created (in the build directory) from the darknet source files ("/src/yoloInterface/darknet") and the darknet makefile. So, to update the darknet library after changing its source files, rebuild the entire project. By default, darknet is built with OpenMP, Cuda, and CUDNN.
