#include <fstream>

#include "functions.h"

//x,y = center - normalized to [0,1] | w,h = width,height - normalized to [0,1] - output format of YOLO

cv::Rect getBoundingBoxYolo(const cv::Mat & img, float x, float y, float w, float h) {
    cv::Point center(x * img.cols, y * img.rows);
    cv::Point offset(img.cols * w / 2, img.rows * h / 2);
    return cv::Rect(center - offset, center + offset);
}

//use for each layer

template<int GRIDS >
struct YoloPredictions {
    //file must be properly formatted

    YoloPredictions(const cv::Mat img, const std::string & fileName) {
        std::ifstream dataFile(fileName);
        assert(dataFile.is_open());
        std::stringstream dataBuffer;
        dataBuffer << dataFile.rdbuf();
        dataFile.close();

        while (!dataBuffer.eof()) {
            int gridR, gridC, anchor;
            char tempChar;
            //get grid and anchor number
            dataBuffer >> gridR >> tempChar >> gridC >> tempChar >> anchor >> tempChar;

            //get (x,y)(w,h)
            float x, y, w, h;
            dataBuffer >> tempChar >> x >> tempChar >> y >> tempChar >> tempChar >> w >> tempChar >> h >> tempChar;

            //get object score
            float objScore;
            dataBuffer >> tempChar >> objScore >> tempChar;

            addPrediction(gridR, gridC, anchor, getBoundingBoxYolo(img, x, y, w, h), objScore);
        }
    }

    std::pair<cv::Rect, float> getPrediction(unsigned int r, unsigned int c, unsigned int anchor) const {
        return std::pair<cv::Rect, float> (m_predictions[r][c][anchor], m_scores[r][c][anchor]);
    }

    void addPrediction(unsigned int r, unsigned int c, unsigned int anchor, const cv::Rect & rect, float objScore) {
        m_predictions[r][c][anchor] = rect;
        m_scores[r][c][anchor] = objScore;
    }

    cv::Rect m_predictions[GRIDS][GRIDS][3]; //predictions - [row][col][b]
    float m_scores[GRIDS][GRIDS][3]; //object-ness score - [row][col][b]
};

//only supports 3 layers

template<int LAYERSIZE1, int LAYERSIZE2, int LAYERSIZE3>
void interactiveDisplay(const cv::Mat & img, const YoloPredictions<LAYERSIZE1> & p1, const YoloPredictions<LAYERSIZE2> & p2, YoloPredictions<LAYERSIZE3> & p3) {
    constexpr int layerSizes[3] = {LAYERSIZE1, LAYERSIZE2, LAYERSIZE3};
    const cv::Scalar colors[3] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
    std::pair<cv::Rect, float> predictions[3];
    cv::Mat dispImg;
    unsigned int r = 0, c = 0, l = 0; //display row, col, level

    std::string winName = "YOLO output", title;
    DisplayImg(img, winName, 800, 800);

    while (true) {
        dispImg = img.clone();

        DrawGridLines(dispImg, layerSizes[l]);

        if (l == 0)
            for (unsigned int a = 0; a < 3; ++a)
                predictions[a] = p1.getPrediction(r, c, a);
        else if (l == 1)
            for (unsigned int a = 0; a < 3; ++a)
                predictions[a] = p2.getPrediction(r, c, a);
        else if (l == 2)
            for (unsigned int a = 0; a < 3; ++a)
                predictions[a] = p3.getPrediction(r, c, a);
        else
            assert(0); //error - should not be possible

        for (unsigned int a = 0; a < 3; ++a) {
            DrawBoundingBox(dispImg, predictions[a].first, colors[a]);
            cv::putText(dispImg, std::to_string(predictions[a].second), cv::Point(10, 25 + 25 * a), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, colors[a]);
        }

        title = "Level: " + std::to_string(l) + " (r=" + std::to_string(r) + "/" + std::to_string(layerSizes[l] - 1) + ",c=" + std::to_string(c) + "/" + std::to_string(layerSizes[l] - 1) + ")";
        char key = UpdateImg(dispImg, winName, title, true);

        if (key == 'q' || key == 'c' || key == 033)
            break; //exit
        else if (key == '1')
            l = 0, r = std::min((int) r, layerSizes[l] - 1), c = std::min((int) c, layerSizes[l] - 1);
        else if (key == '2')
            l = 1, r = std::min((int) r, layerSizes[l] - 1), c = std::min((int) c, layerSizes[l] - 1);
        else if (key == '3')
            l = 2;
        else if (key == 'w' && r > 0)
            --r;
        else if (key == 'a' && c > 0)
            --c;
        else if (key == 's' && r < layerSizes[l] - 1)
            ++r;
        else if (key == 'd' && c < layerSizes[l] - 1)
            ++c;
    }

    cv::destroyWindow(winName);
}

int main(int argc, char * argv[]) {
    cv::Mat img = cv::imread("/home/dp/Desktop/ARL/darknet-master/networkInput.jpg");

    const int imgSize = 608; //fixed input size of yolo network image
    constexpr int predictionSizes[3] = {19, 38, 76}; //size of prediction layers
    const std::string file19 = "/home/dp/Desktop/ARL/darknet-master/Layer_82_output.txt";
    const std::string file38 = "/home/dp/Desktop/ARL/darknet-master/Layer_94_output.txt";
    const std::string file76 = "/home/dp/Desktop/ARL/darknet-master/Layer_106_output.txt";
    assert(img.rows == imgSize && img.cols == img.cols);

    YoloPredictions < predictionSizes[0] > layer82(img, file19);
    YoloPredictions < predictionSizes[1] > layer94(img, file38);
    YoloPredictions < predictionSizes[2] > layer106(img, file76);

    interactiveDisplay(img, layer82, layer94, layer106);

    return 0;
}