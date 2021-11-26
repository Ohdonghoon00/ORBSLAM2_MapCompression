#pragma once

#include <vector>
#include "System.h"
#include <map>
#include "BoostArchiver.h"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>




class DataBase
{
public:    
    
    DataBase()
    {}
    
    // Map info
    std::map< int, cv::Point3f > Landmarks;
    std::vector<cv::Mat> Descriptors;

    // Keyframe info
    std::map< int, std::vector< int > > KFtoMPIdx; // KFid - MPidx 
    std::vector<double> timestamps;




    // for save/load 
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version);
    
    // get info
    cv::Mat GetKFMatDescriptor(int idx);
    std::vector<cv::Point3f> GetKF3dPoint(int idx);





};