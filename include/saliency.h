#ifndef SALIENCY_H
#define SALIENCY_H

#include <opencv2/core.hpp>

namespace SaliencyFilter {

    void removeUnsalient(const cv::Mat &img, const std::vector<cv::Rect> &segmentation_regions, const std::vector<float> &segmentation_scores, std::vector<cv::Rect> &surviving_regions);
    void calculateMeanSTD(const std::vector<float> &data, float &mean, float &stdeviation);

    class LineDescriptor {
    public:
        LineDescriptor(void);
        ~LineDescriptor(void);

        //LineDescriptor is not meant to be copied - but can be moved
        LineDescriptor(const LineDescriptor&) = delete;
        LineDescriptor& operator=(const LineDescriptor&) = delete;
        LineDescriptor& operator=(LineDescriptor&&) = default;
        LineDescriptor(LineDescriptor&&) = default;

    public:
        void compute(const cv::Mat &saliency_map, bool is_horizontal, const cv::Point &edge_start, unsigned int edge_length, int box_size, unsigned int expansion_length_positive, unsigned int expansion_length_negative);

        void resize(int min, int max);

        float& operator[](int position);
        const float& operator[](int position) const;

        int getMin(void) const;
        int getMax(void) const;

        cv::Point getStart(void) const;
        cv::Point getEnd(void) const;

        int getCenter(void) const;

        void scaleData(float factor, bool divide = false);

        void visualizeDescriptor(const cv::Mat &saliency_map) const;

        int computeOptimalChange(void) const;

        bool hasPointOfInflection(void) const;

    private:
        cv::Point m_start;
        unsigned int m_length;
        bool m_horizontal;

        float *m_data;
        float *m_derivative;
        float m_derivative_avg;
        float *m_derivative_2;
        int m_size;
        int m_min;
        int m_max;
    };

    enum RECT_SIDES {
        LEFT = 0,
        RIGHT = 1,
        TOP = 2,
        BOTTOM = 3,
        NUM_SIDES = 4
    };

    class SaliencyAnalyzer {
    public:
        cv::Mat m_img;

        SaliencyAnalyzer(const cv::Mat &img);

        ~SaliencyAnalyzer(void);

        //SaliencyAnalyzer is not meant to be copied or moved - for simplicity
        SaliencyAnalyzer(const SaliencyAnalyzer&) = delete;
        SaliencyAnalyzer(SaliencyAnalyzer&&) = delete;
        SaliencyAnalyzer& operator=(const SaliencyAnalyzer&) = delete;
        SaliencyAnalyzer& operator=(SaliencyAnalyzer&&) = delete;

    public:
        void addSegmentedRegion(const cv::Rect &region, float segmentation_score);

        void mergeSubRegions(float overlap_thresh = 0.92, float min_size_of_inner = 0.5);

        void applySaliencyThreshold(void);

        void computeSaliencyDescriptors(void);

        void getRegionsSurviving(std::vector<cv::Rect> &regions) const;

    private:
        void sortRegionsByArea(void); //from high low
        void removeInvalidRegions(void);

    private:

        struct Region {
            Region(const cv::Mat &saliency_map, const cv::Rect &region, float score, float saliency_map_std);

            void ensureRegionSaliency(float std_thresh_overall, float saliency_mean, float saliency_std);

            void computeDescriptorsAlongAllEdges(const cv::Mat &saliency_map, const cv::Mat &img);

            void computeDescriptorAlongSingleEdge(RECT_SIDES side, const cv::Mat &saliency_map);

            void resizeBoxEdge(RECT_SIDES side, int change);

            cv::Rect box;
            float box_sal;
            float box_num_pixels;

            cv::Rect box_double;
            float box_sal_double;
            float box_num_pixels_double;

            float avg_sal;
            float avg_sal_double; //average saliency of surrounding region - doesn't include original region
            float std_sal;
            float std_sal_double;

            float avg_sal_surroundings;
            float std_change_from_surroundings; //+ = good, - = bad

            float std_saliency_map;

            int status; //1 = fine, 0 = merged with another detection, -1 = removed by standard deviation too low, -2 = removed because too much like neighbors
            float score;
            LineDescriptor edges[RECT_SIDES::NUM_SIDES];
        };

        cv::Mat m_saliency_map;
        float m_saliency_mean;
        float m_saliency_std;

        std::vector<Region> m_regions;
    };
};

#endif
