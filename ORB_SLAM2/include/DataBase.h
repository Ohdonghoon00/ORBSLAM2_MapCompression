#pragma once

#include <vector>
#include "System.h"
#include <map>
#include "BoostArchiver.h"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include "Converter.h"



class DataBase
{
public:    
    
    DataBase()
    {}
    
    // Map info
    std::map< int, cv::Point3d > Landmarks;
    std::map< int, cv::Mat> Descriptors;


    // Keyframe info
    std::map< int, std::vector< int > > KFtoMPIdx; // KFid - MPidx 
    std::map< int, std::vector< cv::KeyPoint > > KeyPointInMap;
    std::vector<double> timestamps;
    std::map< int, Vector6d > kfPoses;

    // Keyframe Img
    std::vector<cv::Mat> LeftKFimg, RightKFimg;

    // for LoopClosure
    std::map< int, cv::Mat > dbow2Descriptors;

    DataBase(const DataBase &tc)
    {
        Landmarks = tc.Landmarks;
        Descriptors = tc.Descriptors;
        KFtoMPIdx = tc.KFtoMPIdx;
        KeyPointInMap = tc.KeyPointInMap;
        timestamps = tc.timestamps;
        LeftKFimg = tc.LeftKFimg;
        RightKFimg = tc.RightKFimg;
        kfPoses = tc.kfPoses;
        dbow2Descriptors = tc.dbow2Descriptors;
    }

    // for save/load 
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version);
    
    // get info
    cv::Mat GetKFMatDescriptor(int idx);
    std::vector<cv::Point3d> GetKF3dPoint(int idx);
    
    // function
    int GetObservationCount(int idx);



};

