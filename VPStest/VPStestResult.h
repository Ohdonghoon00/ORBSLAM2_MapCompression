#pragma once

#include <vector>
#include <map>
#include <set>
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"


#include "BoostArchiver.h"

class VPStestResult
{
public:
    
    VPStestResult()
    {}


    // SolvePnP
    std::map< int, cv::Mat> MatchingImg;
    // std::map< int, Vector6d > PnPpose;
    std::map< int, double > PnPInlierRatio;
    std::map< int, int > PnPInliers;

    // Place Recognition
    std::map< int, int > DBoW2ResultImgNum;


    std::vector<int> KFimageNum;

    // for save/load
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version);
    
};