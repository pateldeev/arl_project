#include <opencv2/core.hpp>

#include "saliency.h"
#include "functions.h"

#include <iomanip>

namespace SaliencyFilter {

    LineDescriptor::LineDescriptor(void) : m_start(0, 0), m_length(0), m_data(nullptr), m_derivative(nullptr), m_second_derivative(nullptr), m_size(0), m_min(0), m_max(-1) {
    }

    LineDescriptor::~LineDescriptor(void) {
        delete [] m_data;
        delete [] m_derivative;
        delete [] m_second_derivative;
    }

    void LineDescriptor::compute(const cv::Mat &saliency_map, bool is_horizontal, const cv::Point &edge_start, unsigned int edge_length, int box_size, unsigned int expansion_length_positive, unsigned int expansion_length_negative) {
        m_start = edge_start;
        m_horizontal = is_horizontal;
        m_length = edge_length;

        if (!m_horizontal) { //line is vertical
            CV_Assert(m_start.x < saliency_map.cols && m_start.y + m_length < saliency_map.rows);

            int x_min = std::max(0, int(m_start.x - expansion_length_negative));
            int x_max = std::min(saliency_map.cols - 1, int(m_start.x + expansion_length_positive));

            bool box_is_left = (box_size < 0);
            cv::Range row_range(m_start.y, m_start.y + m_length + 1);
            cv::Range col_range;
            if (box_is_left)
                col_range.start = m_start.x + box_size;
            else
                col_range.end = m_start.x + box_size + 1;

            resize(x_min - m_start.x, x_max - m_start.x);

            for (int x = x_min; x <= x_max; ++x) {
                if (box_is_left)
                    col_range.end = x + 1;
                else
                    col_range.start = x;
                m_data[x - m_start.x - m_min] = cv::sum(saliency_map(row_range, col_range))[0];
            }

        } else { //line is horizontal
            CV_Assert(m_start.x + edge_length < saliency_map.cols && m_start.y < saliency_map.rows);

            int y_min = std::max(0, int(m_start.y - expansion_length_negative));
            int y_max = std::min(saliency_map.rows - 1, int(m_start.y + expansion_length_positive));

            bool box_is_above = (box_size < 0);
            cv::Range row_range;
            cv::Range col_range(m_start.x, m_start.x + m_length + 1);
            if (box_is_above)
                row_range.start = m_start.y + box_size;
            else
                row_range.end = m_start.y + box_size + 1;

            resize(y_min - m_start.y, y_max - m_start.y);

            for (int y = y_min; y <= y_max; ++y) {
                if (box_is_above)
                    row_range.end = y + 1;
                else
                    row_range.start = y;

                if (y - m_start.y == 1) {
                    std::cout << std::endl << row_range << "," << col_range << "|" << cv::sum(saliency_map(row_range, col_range))[0] << std::endl;
                }

                m_data[y - m_start.y - m_min] = cv::sum(saliency_map(row_range, col_range))[0];
            }
        }

        //compute derivative
        float temp = 2 * (m_length + 1);
        for (int i = 1; i < m_size - 1; ++i)
            m_derivative[i] = std::abs((m_data[i + 1] - m_data[i - 1])) / temp;
        m_derivative[0] = m_derivative[m_size - 1] = 0.f;

        for (int i = 1; i < m_size - 1; ++i)
            m_second_derivative[i] = m_derivative[i + 1] - m_derivative[i - 1];
        m_second_derivative[0] = m_second_derivative[m_size - 1] = 0.f;
    }

    void LineDescriptor::resize(int min, int max) {
        CV_Assert(max >= 2 && min <= -2);
        delete m_data;
        delete m_derivative;
        m_min = min, m_max = max;
        m_size = m_max - m_min + 1;
        m_data = new float[m_size];
        m_derivative = new float[m_size];
        m_second_derivative = new float[m_size];
    }

    float& LineDescriptor::operator[](int position) {
        return m_data[position - m_min];
    }

    const float& LineDescriptor::operator[](int position) const {
        return m_data[position - m_min];
    }

    int LineDescriptor::getMin(void) const {
        return m_min;
    }

    int LineDescriptor::getMax(void) const {
        return m_max;
    }

    cv::Point LineDescriptor::getStart(void) const {
        return m_start;
    }

    cv::Point LineDescriptor::getEnd(void) const {
        return (m_horizontal) ? cv::Point(m_start.x + m_length, m_start.y) : cv::Point(m_start.x, m_start.y + m_length);
    }

    void LineDescriptor::scaleData(float factor, bool divide) {
        for (int i = 0; i < m_size; ++i)
            if (divide)
                m_data[i] /= factor;
            else
                m_data[i] *= factor;
    }

    void LineDescriptor::visualizeDescriptor(const cv::Mat &saliency_map) const {
        int min_pos, max_pos;
        float min_val, max_val;
        if (m_data[0] < m_data[m_size - 1]) {
            min_pos = m_min, max_pos = m_max;
            min_val = m_data[0], max_val = m_data[m_size - 1];
        } else {
            min_pos = m_max, max_pos = m_min;
            min_val = m_data[m_size - 1], max_val = m_data[0];
        }
        std::cout << "Descriptor min|max: " << min_val << "(@" << min_pos << ")|" << max_val << "(@" << max_pos << ")" << std::endl;

        cv::Mat disp = saliency_map.clone();
        if (disp.type() == CV_8UC1)
            cv::cvtColor(disp, disp, cv::COLOR_GRAY2BGR);

        cv::Point end = getEnd();
        cv::line(disp, m_start, end, cv::Scalar(255, 0, 0));
        if (m_horizontal) {
            cv::line(disp, cv::Point(m_start.x, m_start.y + m_min), cv::Point(end.x, end.y + m_min), cv::Scalar(0, 0, 255));
            cv::line(disp, cv::Point(m_start.x, m_start.y + m_max), cv::Point(end.x, end.y + m_max), cv::Scalar(0, 0, 255));
        } else {
            cv::line(disp, cv::Point(m_start.x + m_min, m_start.y), cv::Point(end.x + m_min, end.y), cv::Scalar(0, 0, 255));
            cv::line(disp, cv::Point(m_start.x + m_max, m_start.y), cv::Point(end.x + m_max, end.y), cv::Scalar(0, 0, 255));
        }
        DisplayImg(disp, "Descriptor_Line");

        cv::Mat descriptor_function(100, m_max - m_min + 1, CV_8UC1, cv::Scalar(0, 0, 0));
        for (int r = 0; r < descriptor_function.rows; ++r)
            descriptor_function.at<uint8_t>(r, 0 - m_min) = 125;

        float value;
        int sub;
        for (int i = 0; i < m_size; ++i) {
            value = (m_data[i] / max_val) * 90;
            sub = 1;
            for (int r = 0; r < value && sub <= descriptor_function.rows; ++r, ++sub)
                descriptor_function.at<uint8_t>(descriptor_function.rows - sub, i) = 255;
        }
        DisplayImg(descriptor_function, "Descriptor_Values");

        for (int i = 0; i < m_size; ++i)
            std::cout << "   " << std::right << std::setw(3) << (i + m_min) << ": " << std::right << std::setw(10) << m_data[i] << "   |Diff: " << std::right << std::setw(10) << m_derivative[i] << "   |Diff2: " << m_second_derivative[i] << std::endl;
    }

    int LineDescriptor::computeOptimalChange(float expansion_thresh, float compression_thresh) const {
        bool is_pos_higher = (m_data[m_size - 1] >= m_data[0]);
        int optimal_change = 0;

        while (optimal_change > m_min && optimal_change < m_max) {
            float current_derivative = m_derivative[optimal_change - m_min];
            std::cout << "|" << optimal_change << "|" << current_derivative << " --- ";
            if (current_derivative > expansion_thresh) { //expand
                std::cout << "Expanding";
                (is_pos_higher) ? ++optimal_change : --optimal_change;
            } else if (current_derivative < compression_thresh) { //compress
                std::cout << "Compressing";
                (is_pos_higher) ? --optimal_change : ++optimal_change;
            } else {
                break;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        return optimal_change;
    }
};