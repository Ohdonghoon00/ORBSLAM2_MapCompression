#pragma once

#include<opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>

#include "DataBase.h"
#include "Converter.h"
#include "Parameter.h"
#include "ORBVocabulary.h"
#include "utils.h"

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

extern cv::Mat K;

class VPStest
{
public:
    
    VPStest(int num):
        FeatureNum(num)
    {}

    int FeatureNum;

    cv::Ptr<cv::ORB> orb = cv::ORB::create(FeatureNum);
    std::map<int, std::vector<cv::Point3f>> MatchDB3dPoints;
    std::map<int, std::vector<cv::Point2f>> MatchQ2dPoints;
    std::vector<int> CandidateKFid;

    void SetCandidateKFid(DBoW2::QueryResults ret);

    int FindReferenceKF(DataBase* DB, cv::Mat QDescriptor, std::vector<cv::KeyPoint> QKeypoints);
    double PnPInlierRatio(int KFid);
    double VPStestToReferenceKF(DataBase* DB, cv::Mat QDescriptor, std::vector<cv::KeyPoint> QKeypoints, int KFid, Eigen::Matrix4d &Pose, cv::Mat &Inliers, std::vector<cv::DMatch> &GoodMatches_);

    cv::Mat InputQueryImg(std::string QueryFile);
    std::vector<cv::KeyPoint> ORBFeatureExtract(cv::Mat img);
    cv::Mat ORBDescriptor(cv::Mat img, std::vector<cv::KeyPoint> keypoints);
    std::vector<cv::DMatch> ORBDescriptorMatch(cv::Mat trainDescriptor, cv::Mat queryDescriptor);
    int FindKFImageNum(int KFid, DataBase* DB, std::vector<double> timestamps);
    void InlierMatchResult(std::vector<cv::DMatch> &Matches, cv::Mat Inliers);


};



