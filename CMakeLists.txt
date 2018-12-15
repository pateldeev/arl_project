PROJECT(Projects)
CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

SET (CMAKE_CXX_STANDARD 11)

FIND_PACKAGE(OpenCV 4.0.0 REQUIRED)

INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include" ${OpenCV_INCLUDE_DIRS} )

ADD_EXECUTABLE(regions ${PROJECT_SOURCE_DIR}/src/regions.cpp ${PROJECT_SOURCE_DIR}/src/functions.cpp)
ADD_EXECUTABLE(saliency ${PROJECT_SOURCE_DIR}/src/saliency.cpp ${PROJECT_SOURCE_DIR}/src/functions.cpp)
ADD_EXECUTABLE(filteredRegions ${PROJECT_SOURCE_DIR}/src/filteredRegions.cpp ${PROJECT_SOURCE_DIR}/src/functions.cpp)

ADD_EXECUTABLE(testarea ${PROJECT_SOURCE_DIR}/src/test.cpp ${PROJECT_SOURCE_DIR}/src/functions.cpp ${PROJECT_SOURCE_DIR}/src/segmentation/selectiveSegmentation.cpp)
ADD_EXECUTABLE(contours ${PROJECT_SOURCE_DIR}/src/contours.cpp)

ADD_EXECUTABLE(yoloVisualization ${PROJECT_SOURCE_DIR}/src/yoloInterface/main.cpp)

TARGET_LINK_LIBRARIES(regions ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(saliency ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(filteredRegions ${OpenCV_LIBS})

TARGET_LINK_LIBRARIES(testarea ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(contours ${OpenCV_LIBS})

TARGET_LINK_LIBRARIES(yoloVisualization ${OpenCV_LIBS})

