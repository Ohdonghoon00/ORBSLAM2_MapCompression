#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <map>

class Map
{
public:

    Map()
    {}

    

    // Landmark
    std::map< int, cv::Point3d > Map3dpts;
    // std::vector< int > MapIds;
    
        
    std::map< int, cv::Mat> MapDesctriptors;
    
    // Keyframe
    std::map< int, cv::Mat > dbow2Descriptors; // for dbow2 LoopClosure
    std::map< int, cv::Mat > KFimg;
    std::map< int, cv::KeyPoint > KF2dptsInMap;
    std::map< int, std::vector< int > > KFtoMPIdx; // KFid - MPidx
};