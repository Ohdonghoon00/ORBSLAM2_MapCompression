#pragma once

#include<opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>

#include "DataBase.h"
#include "Converter.h"
#include "Parameter.h"
#include "ORBVocabulary.h"

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>



class VPStest
{
public:
    
    VPStest()
    {}

    cv::Ptr<cv::ORB> orb = cv::ORB::create(4000);
    std::map<int, std::vector<cv::Point3f>> MatchDB3dPoints;
    std::map<int, std::vector<cv::Point2f>> MatchQ2dPoints;
    std::vector<int> CandidateKFid;

    void SetCandidateKFid(DBoW2::QueryResults ret);
    // void SetCandidateKF3dpointDB(DataBase DB);

    // void DescriptorMatch(DataBase DB, cv::Mat QDescriptor);
    int FindReferenceKF(DataBase DB, cv::Mat QDescriptor, std::vector<cv::KeyPoint> QKeypoints);
    double PnPInlierRatio(int KFid);
    double VPStestToReferenceKF(DataBase DB, cv::Mat QDescriptor, std::vector<cv::KeyPoint> QKeypointsint, int KFid, Eigen::Matrix4f &Pose);

    cv::Mat InputQueryImg(std::string QueryFile);
    std::vector<cv::KeyPoint> ORBFeatureExtract(cv::Mat img);
    cv::Mat ORBDescriptor(cv::Mat img, std::vector<cv::KeyPoint> keypoints);
    std::vector<cv::DMatch> ORBDescriptorMatch(cv::Mat trainDescriptor, cv::Mat queryDescriptor);

    // Converter
    // cv::Mat VectorMatToMat(std::vector<cv::Mat> Descriptor);
    // std::vector<cv::Mat> MatToVectorMat(cv::Mat Descriptor);

    // std::vector<cv::Point3f> Sort3dpointByMatch(std::vector<cv::Point3f> Disorder3dpoint, std::vector<cv::DMatch> matches);


};

