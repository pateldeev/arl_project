#ifndef SELECTIVESEGMENTATION_H
#define SELECTIVESEGMENTATION_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/ximgproc/segmentation.hpp>

namespace Segmentation {

    void process(const cv::Mat & img, std::vector<cv::Rect> & rects, int base_k = 150, int inc_k = 150, float sigma = 0.8);

    // Represent a region

    class Region {
    public:
        int id;
        int level;
        int merged_to;
        double rank;
        cv::Rect bounding_box;

        Region(void) : id(0), level(0), merged_to(0), rank(0) {
        }

        friend std::ostream& operator<<(std::ostream & os, const Region & n);

        bool operator<(const Region & n) const {
            return rank < n.rank;
        }
    };

    // Comparator to sort cv::rect (used for a std::map).

    struct rectComparator {

        bool operator()(const cv::Rect & a, const cv::Rect & b) const {
            if (a.x < b.x)
                return true;
            else if (a.x > b.x)
                return false;

            if (a.y < b.y)
                return true;
            else if (a.y > b.y)
                return false;

            if (a.width < b.width)
                return true;
            else if (a.width > b.width)
                return false;

            if (a.height < b.height)
                return true;
            else if (a.height > b.height)
                return false;

            return false;
        }
    };

    // Represent a neighbor

    class Neighbor {
    public:
        int from;
        int to;
        float similarity;
        friend std::ostream & operator<<(std::ostream & os, const Neighbor & n);

        bool operator<(const Neighbor & n) const {
            return similarity < n.similarity;
        }
    };
};

#endif