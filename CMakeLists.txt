PROJECT(ARL_Project)
CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

SET(CMAKE_CXX_STANDARD 17)

#Required packages/libraries
FIND_PACKAGE(OpenCV 4.0.0 REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)

SET(INCLUDES ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src/yoloInterface/darknet/include ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
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

#create various executables
ADD_EXECUTABLE(regions "${PROJECT_SOURCE_DIR}/src/regions.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp")
ADD_EXECUTABLE(saliency "${PROJECT_SOURCE_DIR}/src/saliency.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp")
ADD_EXECUTABLE(filteredRegions "${PROJECT_SOURCE_DIR}/src/filteredRegions.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp")

ADD_EXECUTABLE(testarea "${PROJECT_SOURCE_DIR}/src/test.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp" "${PROJECT_SOURCE_DIR}/src/segmentation/selectiveSegmentation.cpp")
ADD_EXECUTABLE(contours "${PROJECT_SOURCE_DIR}/src/contours.cpp")

ADD_EXECUTABLE(yoloVisualization "${PROJECT_SOURCE_DIR}/src/yoloInterface/visualization.cpp")
ADD_EXECUTABLE(yolo "${PROJECT_SOURCE_DIR}/src/yoloInterface/main.cpp" "${PROJECT_SOURCE_DIR}/src/functions.cpp" "${PROJECT_SOURCE_DIR}/src/yoloInterface/yoloInterface.cpp")

#link executables to libraries
TARGET_LINK_LIBRARIES(regions ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(saliency ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(filteredRegions ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(testarea ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(contours ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(yoloVisualization ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(yolo ${OpenCV_LIBS} ${DARKNET_LIBS})