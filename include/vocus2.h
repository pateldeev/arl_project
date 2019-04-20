/*****************************************************************************
 *
 * VOCUS2.h file for the saliency program VOCUS2. 
 * A detailed description of the algorithm can be found in the paper: "Traditional Saliency Reloaded: A Good Old Model in New Shape", S. Frintrop, T. Werner, G. Martin Garcia, in Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2015.  
 * Please cite this paper if you use our method.
 *
 * Implementation:	  Thomas Werner   (wernert@cs.uni-bonn.de)
 * Design and supervision: Simone Frintrop (frintrop@iai.uni-bonn.de)
 *
 * Version 1.1
 *
 * This code is published under the MIT License 
 * (see file LICENSE.txt for details)
 *
 ******************************************************************************/

#ifndef VOCUS2_H_
#define VOCUS2_H_

#include <opencv2/core/core.hpp>

#include <string>
#include <fstream>

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/assume_abstract.hpp>

// different colorspaces

enum ColorSpace {
    // CIELab
    LAB = 0,
    OPPONENT_CODI = 1, // like in Klein/Frintrop DAGM 2012
    OPPONENT = 2, // like above but shifted and scaled to [0,1]
    // splitted RG and BY channels
    ITTI = 3
};

// fusing operation to build the feature, conspicuity and saliency map(s)

enum FusionOperation {
    ARITHMETIC_MEAN = 0,
    MAX = 1,
    // uniqueness weight as in 
    // Simone Frintrop: VOCUS: A Visual Attention System for Object Detection and Goal-directed Search, PhD thesis 2005
    UNIQUENESS_WEIGHT = 2,
};

// pyramid structure

enum PyrStructure {
    // two independent pyramids
    CLASSIC = 0,
    // two independent pyramids derived from a base pyramid
    CODI = 1,
    // surround pyramid derived from center pyramid
    NEW = 2,
    // single pyramid (iNVT)
    SINGLE = 3
};

// class containing all parameters for the main class

class VOCUS2_Cfg {
public:
    // default constructor, default parameters

    VOCUS2_Cfg(void) {
        c_space = OPPONENT_CODI;
        fuse_feature = ARITHMETIC_MEAN;
        fuse_conspicuity = ARITHMETIC_MEAN;
        start_layer = 0;
        stop_layer = 4;
        center_sigma = 3;
        surround_sigma = 13;
        n_scales = 2;
        normalize = true;
        pyr_struct = NEW;
        orientation = false;
        combined_features = false;
    };

    // constuctor for a given config file

    VOCUS2_Cfg(std::string f_name) {
        load(f_name);
    }

    virtual ~VOCUS2_Cfg() {
    };


    ColorSpace c_space;
    FusionOperation fuse_feature, fuse_conspicuity;
    PyrStructure pyr_struct;

    int start_layer, stop_layer, n_scales;
    float center_sigma, surround_sigma;

    bool normalize, orientation, combined_features;

    // load xml file

    bool load(std::string f_name) {
        std::ifstream conf_file(f_name);
        if (conf_file.good()) {
            boost::archive::xml_iarchive ia(conf_file);
            ia >> boost::serialization::make_nvp("VOCUS2_Cfg", *this);
            conf_file.close();
            return true;
        } else std::cout << "Config file: " << f_name << " not found. Using defaults." << std::endl;
        return false;
    }

    // wite to xml file

    bool save(std::string f_name) {
        std::ofstream conf_file(f_name);
        if (conf_file.good()) {
            boost::archive::xml_oarchive oa(conf_file);
            oa << boost::serialization::make_nvp("VOCUS2_Cfg", *this);
            conf_file.close();
            return true;
        }
        return false;
    }

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP(c_space);
        ar & BOOST_SERIALIZATION_NVP(fuse_feature);
        ar & BOOST_SERIALIZATION_NVP(fuse_conspicuity);
        ar & BOOST_SERIALIZATION_NVP(pyr_struct);
        ar & BOOST_SERIALIZATION_NVP(start_layer);
        ar & BOOST_SERIALIZATION_NVP(stop_layer);
        ar & BOOST_SERIALIZATION_NVP(center_sigma);
        ar & BOOST_SERIALIZATION_NVP(surround_sigma);
        ar & BOOST_SERIALIZATION_NVP(n_scales);
        ar & BOOST_SERIALIZATION_NVP(normalize);
    }
};

class VOCUS2 {
public:
    VOCUS2(void);
    VOCUS2(const VOCUS2_Cfg& cfg);
    virtual ~VOCUS2(void);

    void setCfg(const VOCUS2_Cfg& cfg);
    // computes center surround contrast on the pyramids
    // does not produce the final saliency map
    // has to be called first
    void process(const cv::Mat& image);

    // add a center bias to the final saliency map
    cv::Mat add_center_bias(float lambda);

    // computes the final saliency map given that process() was called
    cv::Mat get_salmap(void);

    // computes a saliency map for each layer of the pyramid
    std::vector<cv::Mat> get_splitted_salmap(void);

    // write all intermediate results to the given directory
    void write_out(std::string dir);

private:
    VOCUS2_Cfg cfg;
    cv::Mat input;

    cv::Mat salmap;
    std::vector<cv::Mat> salmap_splitted, planes;

    // vectors to hold contrast pyramids as arrays
    std::vector<cv::Mat> on_off_L, off_on_L;
    std::vector<cv::Mat> on_off_a, off_on_a;
    std::vector<cv::Mat> on_off_b, off_on_b;

    // vector to hold the gabor pyramids
    std::vector<std::vector<cv::Mat>> gabor;

    // vectors to hold center and surround gaussian pyramids
    std::vector<std::vector<cv::Mat>> pyr_center_L, pyr_surround_L;
    std::vector<std::vector<cv::Mat>> pyr_center_a, pyr_surround_a;
    std::vector<std::vector<cv::Mat>> pyr_center_b, pyr_surround_b;

    // vector to hold the edge (laplace) pyramid
    std::vector<std::vector<cv::Mat>> pyr_laplace;

    bool salmap_ready, splitted_ready, processed;

    // process image wrt. the desired pyramid structure
    void pyramid_classic(const cv::Mat& image);
    void pyramid_new(const cv::Mat& image);
    void pyramid_codi(const cv::Mat& image);
    // void pyramid_itti(const cv::Mat& image);

    // converts the image to the destination colorspace
    // and splits the color channels
    std::vector<cv::Mat> prepare_input(const cv::Mat& img);

    // clear all datastructures from previous results
    void clear(void);

    // build a multi scale representation based on [Lowe2004]
    std::vector<std::vector<cv::Mat>> build_multiscale_pyr(cv::Mat& img, float sigma = 1.f);

    // combines a vector of Mats into a single mat
    cv::Mat fuse(std::vector<cv::Mat> mat_array, FusionOperation op);

    // computes the center surround contrast
    // uses pyr_center_L
    void center_surround_diff(void);
    void orientation(void);

    // computes the uniqueness of a map by counting the local maxima
    float compute_uniqueness_weight(cv::Mat& map, float t);

    // void mark_equal_neighbours(int r, int c, float value, cv::Mat& map, cv::Mat& marked);
};

#endif