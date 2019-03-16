PROJECT(ARL_Project)
CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

SET(CMAKE_CXX_STANDARD 17)

#Required packages/libraries
FIND_PACKAGE(OpenCV 4.0.0 REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)

#Find flycapture library
FIND_LIBRARY(FlyCapture2_LIBS NAMES flycapture libflycapture)
FIND_PATH(FlyCapture2_INCLUDE_DIRS NAMES "flycapture/FlyCapture2.h" "FlyCapture2.h" PATHS "usr/include" "usr")
if(NOT FlyCapture2_LIBS)
  MESSAGE(FATAL_ERROR "Could not find flycapture library!")
endif(NOT FlyCapture2_LIBS)
if(NOT FlyCapture2_INCLUDE_DIRS)
  MESSAGE(FATAL_ERROR "Could not find flycapture include directory!")
else(NOT FlyCapture2_INCLUDE_DIRS)
  SET(FlyCapture2_INCLUDE_DIRS "${FlyCapture2_INCLUDE_DIRS}/flycapture")
endif(NOT FlyCapture2_INCLUDE_DIRS)

SET(INCLUDES ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src/yoloInterface/darknet/include ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${FlyCapture2_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${INCLUDES})

#Copy darknet (YOLOv3) files to build directory - needed to build darknet library
FILE(COPY "${PROJECT_SOURCE_DIR}/src/yoloInterface/darknet/include/" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/darknet/include")
FILE(COPY "${PROJECT_SOURCE_DIR}/src/yoloInterface/darknet/src/" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/darknet/src")
FILE(COPY "${PROJECT_SOURCE_DIR}/src/yoloInterface/darknet/examples/" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/darknet/examples")
FILE(COPY "${PROJECT_SOURCE_DIR}/src/yoloInterface/darknet/Makefile" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/darknet")

#Build darknet library from source and makefile in "/yoloInterface/darknet"
ADD_DEFINITIONS("-DOPENMP -DGPU -DCUDNN") #options for darknet (originally in makefile)
ADD_CUSTOM_TARGET(DARKNET_BUILD COMMAND make OPENMP=1 GPU=1 CUDNN=1 -j WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/darknet" COMMENT "Building Darknet (YOLOv3)")
ADD_LIBRARY(DARKNET_LIBRARY STATIC IMPORTED)
SET_TARGET_PROPERTIES(DARKNET_LIBRARY PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/darknet/libdarknet.a")
TARGET_LINK_LIBRARIES(DARKNET_LIBRARY INTERFACE pthread ${OpenMP_C_FLAGS} ${CUDA_LIBRARIES} -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand -lcudnn)
ADD_DEPENDENCIES(DARKNET_LIBRARY DARKNET_BUILD)
SET(DARKNET_LIBS DARKNET_LIBRARY)

##source files
SET(SRC_HELPER "${PROJECT_SOURCE_DIR}/src/functions.cpp")
SET(SRC_SEGMENTATION "${PROJECT_SOURCE_DIR}/src/segmentation/segmentation.cpp")
SET(SRC_SEGMENTATION_CV "${PROJECT_SOURCE_DIR}/src/segmentation/segmentationCV.cpp" )
SET(SRC_SALIENCY "${PROJECT_SOURCE_DIR}/src/saliency/saliency.cpp" "${PROJECT_SOURCE_DIR}/src/saliency/SaliencyAnalyzer.cpp" "${PROJECT_SOURCE_DIR}/src/saliency/LineDescriptor.cpp")

#create various executables
#ADD_EXECUTABLE(old_regions "${PROJECT_SOURCE_DIR}/src/old/regions.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp")
#ADD_EXECUTABLE(old_saliency "${PROJECT_SOURCE_DIR}/src/old/saliency.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp")
#ADD_EXECUTABLE(old_filteredRegions "${PROJECT_SOURCE_DIR}/src/old/filteredRegions.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp")
#ADD_EXECUTABLE(old_contours "${PROJECT_SOURCE_DIR}/src/old/contours.cpp")

#ADD_EXECUTABLE(yolo_visualization "${PROJECT_SOURCE_DIR}/src/yoloInterface/visualization.cpp")
#ADD_EXECUTABLE(yolo "${PROJECT_SOURCE_DIR}/src/yoloInterface/yoloMain.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp" "${PROJECT_SOURCE_DIR}/src/yoloInterface/yoloInterface.cpp")

#ADD_EXECUTABLE(webcam "${PROJECT_SOURCE_DIR}/src/webCam.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp" "${PROJECT_SOURCE_DIR}/src/yoloInterface/yoloInterface.cpp" "${PROJECT_SOURCE_DIR}/src/segmentation/segmentationCV.cpp")
ADD_EXECUTABLE(fly_capture_camera "${PROJECT_SOURCE_DIR}/src/flyCamera.cpp" "${PROJECT_SOURCE_DIR}/src/yoloInterface/yoloInterface.cpp" ${SRC_HELPER} ${SRC_SEGMENTATION} ${SRC_HELPER} ${SRC_SALIENCY})
#ADD_EXECUTABLE(fly_capture_camera_detailed "${PROJECT_SOURCE_DIR}/src/flyCameraDetailed.cpp" "${PROJECT_SOURCE_DIR}/src/yoloInterface/yoloInterface.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp" "${PROJECT_SOURCE_DIR}/src/segmentation/segmentation.cpp")
ADD_EXECUTABLE(video "${PROJECT_SOURCE_DIR}/src/video.cpp" ${SRC_HELPER} ${SRC_SEGMENTATION})

#ADD_EXECUTABLE(cv_segmentation "${PROJECT_SOURCE_DIR}/src/segmentation/segMainCV.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp" "${PROJECT_SOURCE_DIR}/src/segmentation/segmentationCV.cpp")
ADD_EXECUTABLE(segmentation "${PROJECT_SOURCE_DIR}/src/segmentation/segMain.cpp" ${SRC_HELPER} ${SRC_SEGMENTATION})
#ADD_EXECUTABLE(segmentation_old "${PROJECT_SOURCE_DIR}/src/segmentation/segMainOld.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp" "${PROJECT_SOURCE_DIR}/src/segmentation/segmentation.cpp")

ADD_EXECUTABLE(test_area "${PROJECT_SOURCE_DIR}/src/test.cpp" ${SRC_HELPER} ${SRC_SEGMENTATION} ${SRC_SALIENCY})
#ADD_EXECUTABLE(test_saliency_merging "${PROJECT_SOURCE_DIR}/src/testSaliency.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp" "${PROJECT_SOURCE_DIR}/src/segmentation/segmentation.cpp")

#link executables to libraries
#TARGET_LINK_LIBRARIES(old_regions ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(old_saliency ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(old_filteredRegions ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(old_contours ${OpenCV_LIBS})

#TARGET_LINK_LIBRARIES(yolo_visualization ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(yolo ${OpenCV_LIBS} ${DARKNET_LIBS})

#TARGET_LINK_LIBRARIES(webcam ${OpenCV_LIBS} ${DARKNET_LIBS})
TARGET_LINK_LIBRARIES(fly_capture_camera ${OpenCV_LIBS} ${FlyCapture2_LIBS} ${DARKNET_LIBS})
#TARGET_LINK_LIBRARIES(fly_capture_camera_detailed ${OpenCV_LIBS} ${FlyCapture2_LIBS} ${DARKNET_LIBS})
TARGET_LINK_LIBRARIES(video ${OpenCV_LIBS} ${DARKNET_LIBS})

#TARGET_LINK_LIBRARIES(cv_segmentation ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(segmentation ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(segmentation_old ${OpenCV_LIBS})

TARGET_LINK_LIBRARIES(test_area ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(test_saliency_merging ${OpenCV_LIBS})