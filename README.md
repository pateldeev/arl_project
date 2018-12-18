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
  * For viewing saliency model outputs
* ./contours
  * For viewing OpenCV contour seqmentation algorithm
* ./filteredRegions
  * For testing out region filtration methodology
* ./testarea
  * For testing experimental stuff - currently tests my own implimentation of [selective segmentation](https://www.robots.ox.ac.uk/~vgg/rg/papers/sande_iccv11.pdf)
* ./yoloVisualization
  * For visualializing the results of created by [YOLOv3](https://github.com/pateldeev/editedYOLOv3)

