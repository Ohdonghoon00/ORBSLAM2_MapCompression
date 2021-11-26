#pragma once

#include <vector>
#include <map>
#include "BoostArchiver.h"




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



    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version);

   
    
    cv::Mat GetKFMatDescriptor(int idx);
    std::vector<cv::Point3f> GetKF3dPoint(int idx);



};

