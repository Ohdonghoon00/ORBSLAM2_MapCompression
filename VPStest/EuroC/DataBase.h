#pragma once

#include <vector>
#include <map>
#include <set>
#include "BoostArchiver.h"




class DataBase
{
public:    
    DataBase()
    {}
    

    // Map info
    std::map< int, cv::Point3f > Landmarks;
    std::map< int, cv::Mat> Descriptors;


    // Keyframe info
    std::map< int, std::vector< int > > KFtoMPIdx; // KFid - MPidx
    std::map< int, std::vector< cv::KeyPoint > > KeyPointInMap; 
    std::vector<double> timestamps;

    // Keyframe Img
    std::vector<cv::Mat> LeftKFimg, RightKFimg;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version);

   


    cv::Mat GetKFMatDescriptor(int idx);
    std::vector<cv::Point3f> GetKF3dPoint(int idx);
    std::vector<cv::KeyPoint> GetKF2dPoint(int idx);
    cv::Mat GetNearReferenceKFMatDescriptor(int rkidx, int nearRkN);
    std::vector<cv::Point3f> GetNearReferenceKF3dPoint(int rkidx, int nearRkN);

    int GetObservationCount(int idx);




};



