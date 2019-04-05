#include <opencv2/core.hpp>

#include "saliency.h"
#include "functions.h"

#include <iomanip>
#include <opencv2/core/base.hpp>

namespace SaliencyFilter {

    LineDescriptor::LineDescriptor(void) : m_start(0, 0), m_length(0), m_data(nullptr), m_derivative(nullptr), m_derivative_2(nullptr), m_size(0), m_min(0), m_max(-1) {
    }

    LineDescriptor::~LineDescriptor(void) {
        delete [] m_data;
        delete [] m_derivative;
        delete [] m_derivative_2;
    }

    void LineDescriptor::compute(const cv::Mat &saliency_map, bool is_horizontal, const cv::Point &edge_start, unsigned int edge_length, int box_size, unsigned int expansion_length_positive, unsigned int expansion_length_negative) {
        m_start = edge_start;
        m_horizontal = is_horizontal;
        m_length = edge_length;
        CV_Assert(expansion_length_positive < std::abs(box_size) && expansion_length_negative < std::abs(box_size));

        if (!m_horizontal) { //line is vertical
            CV_Assert(m_start.x < saliency_map.cols && m_start.y + m_length < saliency_map.rows);

            int x_min = std::max(0, int(m_start.x - expansion_length_negative));
            int x_max = std::min(saliency_map.cols - 1, int(m_start.x + expansion_length_positive));
            resize(x_min - m_start.x, x_max - m_start.x);

            bool box_is_left = (box_size < 0);
            cv::Range row_range(m_start.y, m_start.y + m_length + 1), col_range;

            int data_index;
            if (box_is_left) { //function grows going to right
                col_range.start = m_start.x + box_size;
                col_range.end = x_min + 1;
                m_data[0] = cv::sum(saliency_map(row_range, col_range))[0]; //compute min value of integral

                for (int x = x_min + 1; x <= x_max; ++x) { //go though and add each new part to integral
                    data_index = x - m_start.x - m_min;
                    col_range.start = x, col_range.end = x + 1;
                    m_data[data_index] = m_data[data_index - 1] + cv::sum(saliency_map(row_range, col_range))[0];
                }
            } else { //function grows going to left
                col_range.start = x_max;
                col_range.end = m_start.x + box_size + 1;
                m_data[m_size - 1] = cv::sum(saliency_map(row_range, col_range))[0]; //compute min value of integral

                for (int x = x_max - 1; x >= x_min; --x) { //go through and add each new part to integral 
                    data_index = x - m_start.x - m_min;
                    col_range.start = x, col_range.end = x + 1;
                    m_data[data_index] = m_data[data_index + 1] + cv::sum(saliency_map(row_range, col_range))[0];
                }
            }
        } else { //line is horizontal
            CV_Assert(m_start.x + edge_length < saliency_map.cols && m_start.y < saliency_map.rows);

            int y_min = std::max(0, int(m_start.y - expansion_length_negative));
            int y_max = std::min(saliency_map.rows - 1, int(m_start.y + expansion_length_positive));
            resize(y_min - m_start.y, y_max - m_start.y);

            bool box_is_above = (box_size < 0);
            cv::Range row_range, col_range(m_start.x, m_start.x + m_length + 1);

            int data_index;
            if (box_is_above) { //function grows going down
                row_range.start = m_start.y + box_size;
                row_range.end = y_min + 1;
                m_data[0] = cv::sum(saliency_map(row_range, col_range))[0]; //compute min value of integral

                for (int y = y_min + 1; y <= y_max; ++y) { //go though and add each new part to integral
                    data_index = y - m_start.y - m_min;
                    row_range.start = y, row_range.end = y + 1;
                    m_data[data_index] = m_data[data_index - 1] + cv::sum(saliency_map(row_range, col_range))[0];
                }
            } else { //function grows going up
                row_range.start = y_max;
                row_range.end = m_start.y + box_size + 1;
                m_data[m_size - 1] = cv::sum(saliency_map(row_range, col_range))[0]; //compute min value of integral

                for (int y = y_max - 1; y >= y_min; --y) { //go though and add each new part to integral
                    data_index = y - m_start.y - m_min;
                    row_range.start = y, row_range.end = y + 1;
                    m_data[data_index] = m_data[data_index + 1] + cv::sum(saliency_map(row_range, col_range))[0];
                }
            }
        }

        //compute 1st derivative
        float temp = 2 * (m_length + 1);
        m_derivative_avg = 0.f;
        for (int i = 1; i < m_size - 1; ++i)
            m_derivative_avg += m_derivative[i] = (m_data[i + 1] - m_data[i - 1]) / temp;
        m_derivative[0] = m_derivative[m_size - 1] = 0.f;
        m_derivative_avg /= (m_size - 2);

        //compute 2nd derivative
        for (int i = 2; i < m_size - 2; ++i)
            m_derivative_2[i] = (m_derivative[i + 1] - m_derivative[i - 1]) / 2;
        m_derivative_2[0] = m_derivative_2[1] = m_derivative_2[m_size - 2] = m_derivative_2[m_size - 1] = 0.f;
    }

    void LineDescriptor::resize(int min, int max) {
        CV_Assert(max >= 2 && min <= -2);
        delete m_data;
        delete m_derivative;
        m_min = min, m_max = max;
        m_size = m_max - m_min + 1;
        m_data = new float[m_size];
        m_derivative = new float[m_size];
        m_derivative_2 = new float[m_size];
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
        int optimal_change = computeOptimalChange();
        std::cout << "Descriptor min|max: " << min_val << "(@" << min_pos << ")|" << max_val << "(@" << max_pos << ")" << std::endl;
        std::cout << "Optimal Change: " << optimal_change << std::endl;

        cv::Mat disp = saliency_map.clone();
        if (disp.type() == CV_8UC1)
            cv::cvtColor(disp, disp, cv::COLOR_GRAY2BGR);

        cv::Point end = getEnd();
        cv::line(disp, m_start, end, cv::Scalar(255, 0, 0));
        if (m_horizontal) {
            cv::line(disp, cv::Point(m_start.x, m_start.y + m_min), cv::Point(end.x, end.y + m_min), cv::Scalar(0, 0, 255));
            cv::line(disp, cv::Point(m_start.x, m_start.y + m_max), cv::Point(end.x, end.y + m_max), cv::Scalar(0, 0, 255));

            cv::line(disp, cv::Point(m_start.x, m_start.y + optimal_change), cv::Point(end.x, end.y + optimal_change), cv::Scalar(255, 0, 255));
        } else {
            cv::line(disp, cv::Point(m_start.x + m_min, m_start.y), cv::Point(end.x + m_min, end.y), cv::Scalar(0, 0, 255));
            cv::line(disp, cv::Point(m_start.x + m_max, m_start.y), cv::Point(end.x + m_max, end.y), cv::Scalar(0, 0, 255));

            cv::line(disp, cv::Point(m_start.x + optimal_change, m_start.y), cv::Point(end.x + optimal_change, end.y), cv::Scalar(255, 0, 255));
        }
        DisplayImg(disp, "Descriptor_Line");

        cv::Mat descriptor_function(100, m_max - m_min + 1, CV_8UC1, cv::Scalar(0, 0, 0));
        for (int r = 0; r < descriptor_function.rows; ++r) {
            descriptor_function.at<uint8_t>(r, 0 - m_min) = 150;
            descriptor_function.at<uint8_t>(r, 0 - m_min + optimal_change) = 100;
        }
        float value;
        int sub;
        for (int i = 0; i < m_size; ++i) {
            value = (m_data[i] - min_val) / (max_val - min_val) * 80 + 10;
            sub = 1;
            for (int r = 0; r < value && sub <= descriptor_function.rows; ++r, ++sub)
                descriptor_function.at<uint8_t>(descriptor_function.rows - sub, i) = 255;
        }
        if (min_pos < max_pos) {
            WriteText(descriptor_function, std::to_string(int(m_data[0])), 0.3, cv::Scalar(0, 0, 0), cv::Point(1, 92));
            WriteText(descriptor_function, std::to_string(int(m_data[m_size - 1])), 0.3, cv::Scalar(255, 0, 0), cv::Point(0.7 * descriptor_function.cols, 8));
        } else {
            WriteText(descriptor_function, std::to_string(int(m_data[0])), 0.3, cv::Scalar(255, 0, 0), cv::Point(1, 8));
            WriteText(descriptor_function, std::to_string(int(m_data[m_size - 1])), 0.3, cv::Scalar(0, 0, 0), cv::Point(0.7 * descriptor_function.cols, 92));
        }
        DisplayImg(descriptor_function, "Descriptor_Values");

        std::cout << "Derivative 1 avg: " << m_derivative_avg << std::endl;
        for (int i = 0; i < m_size; ++i)
            std::cout << "   " << std::right << std::setw(3) << (i + m_min) << ": " << std::right << std::setw(10) << m_data[i] << "   |Diff: " << std::right << std::setw(10) << m_derivative[i] << "   |Diff2: " << m_derivative_2[i] << std::endl;
    }

    int LineDescriptor::computeOptimalChange(void) const {
        if (!hasPointOfInflection())
            return 0;

        int max_pos;
        float max_val = 0.f;

        for (int i = 1; i < m_size - 2; ++i) {
            if (std::abs(m_derivative[i]) > max_val) {
                max_pos = i;
                max_val = std::abs(m_derivative[i]);
            }
        }

        int optimal_pos = max_pos;
        if (m_derivative_avg < 0) { //decrease to encapsulate more information
            while (m_derivative[--optimal_pos] < m_derivative_avg);
        } else { //increase to encapsulate more information
            while (m_derivative[++optimal_pos] > m_derivative_avg);
        }

        return (optimal_pos + m_min);
    }

    bool LineDescriptor::hasPointOfInflection(void) const {
        if (m_derivative_2[2] > 0.f) {
            for (int i = 3; i < m_size - 3; ++i) {
                if (m_derivative_2[i] < 0.f)
                    return true;
            }
        } else {
            for (int i = 3; i < m_size - 3; ++i) {
                if (m_derivative_2[i] > 0.f)
                    return true;
            }
        }
        return false;
    }
};