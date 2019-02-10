#include "selectiveSegmentation.h"

#include <map>
#include <algorithm>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>

namespace Segmentation {

    std::vector<cv::Mat> m_images;
    std::vector<cv::Ptr<cv::ximgproc::segmentation::GraphSegmentation> > m_segmentations;
    std::vector<cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategy> > m_strategies;

#ifdef DEBUG_SEGMENTATION
    std::vector<cv::Mat> debugDisp1, debugDisp2;
#endif

    //helper functions

    void switchToSelectiveSearchQuality(const cv::Mat & base_image, const int base_k, const int inc_k, const float sigma) {
        m_images.clear();
        m_segmentations.clear();
        m_strategies.clear();
        m_images.reserve(5);

        cv::Mat hsv;
        cv::cvtColor(base_image, hsv, cv::COLOR_BGR2HSV);
        m_images.push_back(hsv);

        cv::Mat lab;
        cv::cvtColor(base_image, lab, cv::COLOR_BGR2Lab);
        m_images.push_back(lab);

        cv::Mat I;
        cv::cvtColor(base_image, I, cv::COLOR_BGR2GRAY);
        m_images.push_back(I);

        cv::Mat channel[3];
        cv::split(hsv, channel);
        m_images.push_back(channel[0]);

        cv::split(base_image, channel);
        std::vector<cv::Mat> channel2 = {channel[2], channel[1], I};

        cv::Mat rgI;
        cv::merge(channel2, rgI);
        m_images.push_back(rgI);


        for (int k = base_k; k <= base_k + inc_k * 4; k += inc_k)
            m_segmentations.emplace_back(cv::ximgproc::segmentation::createGraphSegmentation(sigma, float(k)));

        cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategyColor> color = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyColor();
        cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategyFill> fill = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyFill();
        cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategyTexture> texture = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyTexture();
        cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategySize> size = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategySize();
        m_strategies.emplace_back(cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyMultiple(color, fill, texture, size));

        cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategyFill> fill2 = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyFill();
        cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategyTexture> texture2 = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyTexture();
        cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategySize> size2 = cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategySize();
        m_strategies.emplace_back(cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyMultiple(fill2, texture2, size2));

        m_strategies.emplace_back(cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategyFill()); //fill3

        m_strategies.emplace_back(cv::ximgproc::segmentation::createSelectiveSearchSegmentationStrategySize()); //size3
    }

    void hierarchicalGrouping(const cv::Mat & img, cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategy> & s, const cv::Mat & img_regions, const cv::Mat_<uint8_t> & is_neighbor, const cv::Mat_<int32_t> & sizes_, const std::vector<cv::Rect> & bounding_rects, std::vector<Region> & regions, int image_id) {
        cv::Mat sizes = sizes_.clone();
        std::vector<Neighbor> similarities;
        regions.clear();

        /////////////////////////////////////////
        s->setImage(img, img_regions, sizes, image_id);

        // Compute initial similarities
        regions.resize(bounding_rects.size());
        for (unsigned int i = 0; i < bounding_rects.size(); ++i) {
            regions[i].id = i;
            regions[i].level = 1;
            regions[i].merged_to = -1;
            regions[i].bounding_box = bounding_rects[i];

            for (unsigned int j = i + 1; j < bounding_rects.size(); ++j) {
                if (is_neighbor.at<uint8_t>(i, j)) {
                    Neighbor n;
                    n.from = i;
                    n.to = j;
                    n.similarity = s->get(i, j);

                    similarities.push_back(n);
                }
            }
        }

        while (similarities.size() != 0) {
            std::sort(similarities.begin(), similarities.end());

            //for (const Neighbor & s : similarities) std::cout << s << std::endl;

            Neighbor p = similarities.back();
            similarities.pop_back();

            Region region_from = regions[p.from], region_to = regions[p.to];

            Region new_r;
            new_r.id = std::min(region_from.id, region_to.id); // Should be the smallest, working ID
            new_r.level = std::max(region_from.level, region_to.level) + 1;
            new_r.merged_to = -1;
            new_r.bounding_box = region_from.bounding_box | region_to.bounding_box;

            regions.push_back(new_r);

            regions[p.from].merged_to = regions[p.to].merged_to = regions.size() - 1;

            // Merge
            s->merge(region_from.id, region_to.id);

            // Update size
            sizes.at<int32_t>(region_from.id, 0) += sizes.at<int32_t>(region_to.id, 0);
            sizes.at<int32_t>(region_to.id, 0) += sizes.at<int32_t>(region_from.id, 0);

            std::vector<int32_t> local_neighbors;

            for (std::vector<Neighbor>::iterator similarity = similarities.begin(); similarity != similarities.end();) {
                if (similarity->from == p.from || similarity->to == p.from || similarity->from == p.to || similarity->to == p.to) {
                    int from = 0;

                    from = (similarity->from == p.from || similarity->from == p.to) ? similarity->to : similarity->from;

                    bool already_neighbor = false;
                    for (const int32_t & local_neighbor : local_neighbors) {
                        if (local_neighbor == from) {
                            already_neighbor = true;
                            break;
                        }
                    }

                    if (!already_neighbor)
                        local_neighbors.push_back(from);

                    similarity = similarities.erase(similarity);
                } else {
                    ++similarity;
                }
            }

            for (const int32_t & local_neighbor : local_neighbors) {
                Neighbor n;
                n.from = regions.size() - 1;
                n.to = local_neighbor;
                n.similarity = s->get(regions[n.from].id, regions[n.to].id);

                similarities.push_back(n);
            }
        }

        // Compute region's rank
        for (Region & r : regions)
            r.rank = (double(std::rand()) / RAND_MAX) * r.level; // Note: this is inverted from the paper, but we keep the lower region first so it's works
    }

    void process(const cv::Mat & img, std::vector<cv::Rect> & rects, int base_k, int inc_k, float sigma) {
        rects.clear();
#ifdef DEBUG_SEGMENTATION
        debugDisp1.clear();
        debugDisp2.clear();
#endif

        switchToSelectiveSearchQuality(img, base_k, inc_k, sigma);

        std::vector<Region> all_regions;

        int image_id = 0;

        for (std::vector<cv::Mat>::const_iterator image = m_images.begin(); image != m_images.end(); ++image) {
            for (std::vector< cv::Ptr<cv::ximgproc::segmentation::GraphSegmentation> >::iterator gs = m_segmentations.begin(); gs != m_segmentations.end(); ++gs) {

                // Compute initial segmentation
                cv::Mat img_regions;
                (*gs)->processImage(*image, img_regions);

                // Get number of regions
                double min, max;
                cv::minMaxLoc(img_regions, &min, &max);
                const int nb_segs = int(max) + 1;

                // Compute bounding rectangles and neighbors
                std::vector<cv::Rect> bounding_rects(nb_segs);
                std::vector< std::vector<cv::Point> > points(nb_segs);

                cv::Mat_<uint8_t> is_neighbor(cv::Mat::zeros(nb_segs, nb_segs, CV_8UC1));
                cv::Mat_<int32_t> sizes(cv::Mat::zeros(nb_segs, 1, CV_32SC1));

                const int * previous_p, * p;
                for (int i = 0; i < img_regions.rows; ++i) {
                    p = img_regions.ptr<int32_t>(i);

                    for (int j = 0; j < img_regions.cols; ++j) {
                        points[p[j]].push_back(cv::Point(j, i));
                        ++sizes.at<int32_t>(p[j], 0);

                        if (i != 0 && j != 0) {
                            is_neighbor.at<uint8_t>(p[j], p[j - 1]) = is_neighbor.at<uint8_t>(p[j], previous_p[j]) = is_neighbor.at<uint8_t>(p[j], previous_p[j - 1]) = 1;

                            is_neighbor.at<uint8_t>(p[j - 1], p[j]) = is_neighbor.at<uint8_t>(previous_p[j], p[j]) = is_neighbor.at<uint8_t>(previous_p[j - 1], p[j]) = 1;
                        }
                    }
                    previous_p = p;
                }

                for (int seg = 0; seg < nb_segs; ++seg)
                    bounding_rects[seg] = cv::boundingRect(points[seg]);

#ifdef DEBUG_SEGMENTATION
                debugDisp1.push_back(img_regions.clone());
                debugDisp2.push_back(image->clone());
                for (const cv::Rect & r : bounding_rects)
                    cv::rectangle(debugDisp2.back(), r, cv::Scalar(0, 255, 0));
#endif

                for (cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategy> strategy : m_strategies) {
                    std::vector<Region> regions;
                    hierarchicalGrouping(*image, strategy, img_regions, is_neighbor, sizes, bounding_rects, regions, image_id);

                    all_regions.insert(all_regions.end(), regions.begin(), regions.end());
                }

                ++image_id;
            }
        }

        std::sort(all_regions.begin(), all_regions.end());

        std::map < cv::Rect, bool, rectComparator > processed_rect;

        // Remove duplicate in rectangle list
        for (const Region & r : all_regions) {
            if (processed_rect.find(r.bounding_box) == processed_rect.end()) {
                processed_rect[r.bounding_box] = true;
                rects.push_back(r.bounding_box);
            }
        }
    }

    std::ostream& operator<<(std::ostream & os, const Neighbor & n) {
        os << "Neighbor[" << n.from << "->" << n.to << "," << n.similarity << "]";
        return os;
    }

    std::ostream& operator<<(std::ostream & os, const Region & r) {
        os << "Region[WID" << r.id << ", L" << r.level << ", merged to " << r.merged_to << ", R:" << r.rank << ", " << r.bounding_box << "]";
        return os;
    }

    void getDomains(const cv::Mat &img, std::vector<cv::Mat> &img_domains, int resize_h) {
        const int old_h = img.rows, old_w = img.cols;
        const int resize_w = old_w * resize_h / old_h;

        img_domains.resize(6); //6 Domains [BGR, HSV, LAB, I, RGI, YUV]

        //BGR
        cv::resize(img, img_domains[0], cv::Size(resize_w, resize_h));

        //HSV
        cv::cvtColor(img_domains[0], img_domains[1], cv::COLOR_BGR2HSV);

        //LAB
        cv::cvtColor(img_domains[0], img_domains[2], cv::COLOR_BGR2Lab);

        //I - intensity
        cv::cvtColor(img_domains[0], img_domains[3], cv::COLOR_BGR2GRAY);

        //RGI
        cv::Mat img_channels[3];
        cv::split(img_domains[0], img_channels);
        cv::Mat rgi_channels[3] = {img_channels[2], img_channels[1], img_domains[3]};
        cv::merge(rgi_channels, 3, img_domains[4]);

        //YUV
        cv::cvtColor(img_domains[0], img_domains[5], cv::COLOR_BGR2YUV);
    }

    void getSegmentations(const std::vector<cv::Mat> &img_domains, std::vector< std::vector<cv::Mat> > &segmentations, const std::vector<float> &k_vals, float sigma) {
        segmentations.resize(img_domains.size(), std::vector<cv::Mat>(k_vals.size()));

        cv::Ptr<cv::ximgproc::segmentation::GraphSegmentation> gs = cv::ximgproc::segmentation::createGraphSegmentation(sigma);

        for (int i_k = 0; i_k < k_vals.size(); ++i_k) {
            gs->setK(k_vals[i_k]);
            for (int i_d = 0; i_d < img_domains.size(); ++i_d)
                gs->processImage(img_domains[i_d], segmentations[i_d][i_k]);
        }
    }

    void getBoundingBoxes(const cv::Mat &img_seg, std::vector<cv::Rect> &boxes, float max_size) {
        double min, max;
        cv::minMaxLoc(img_seg, &min, &max);
        int num_segs = max + 1;

        const float side_ignore_size = 3; //ignore any regions that touch boundaries of image
        const int max_points = max_size * img_seg.rows * img_seg.cols;
        boxes.clear();

        std::unordered_map< int, std::vector<cv::Point> > seg_points;
        for (int i = 0; i < num_segs; ++i)
            seg_points[i];

        const uint32_t * ptr_seg;
        std::unordered_map< int, std::vector<cv::Point> >::iterator points;

        for (int i = 0; i < img_seg.rows; ++i) {
            ptr_seg = img_seg.ptr<uint32_t>(i);

            for (int j = 0; j < img_seg.cols; ++j) {
                int seg = ptr_seg[j];
                points = seg_points.find(seg);
                if (points != seg_points.end()) {
                    if (i < side_ignore_size || i > img_seg.rows - side_ignore_size || j < side_ignore_size || j > img_seg.cols - side_ignore_size) {
                        seg_points.erase(points); //segment is near the edge - ignore region
                    } else {
                        if (points->second.size() > max_points)
                            seg_points.erase(points);
                        else
                            points->second.push_back(cv::Point(j, i));
                    }
                }
            }
        }

        for (const std::pair< int, std::vector<cv::Point> > &p : seg_points)
            boxes.emplace_back(cv::boundingRect(p.second));
    }

    void calculateSegnificantRegions(const std::vector< std::vector< std::vector<cv::Rect> > > &regions_img, std::vector< std::pair<cv::Rect, float> > &regions_significant, float merge_tresh) {
        regions_significant.clear();

        std::vector<float> weights(regions_img[0].size());
        for (int i = 0; i < weights.size(); ++i)
            weights[i] = 1 / (1 + std::exp(-i)); //weight per segmentation level.

        for (const std::vector< std::vector<cv::Rect> > &regions_domain : regions_img) { //iterate through each domain
            for (int s = 0; s < regions_domain.size(); ++s) {
                for (const cv::Rect &region : regions_domain[s]) {
                    bool merged = false;

                    for (std::pair<cv::Rect, float> &r_s : regions_significant) {
                        if (float((region & r_s.first).area()) / (region | r_s.first).area() >= merge_tresh) {
                            r_s.first = region | r_s.first;
                            r_s.second += weights[s];
                            merged = true;
                        }
                    }

                    if (!merged)
                        regions_significant.emplace_back(region, weights[s]);
                }
            }
        }
    }
};