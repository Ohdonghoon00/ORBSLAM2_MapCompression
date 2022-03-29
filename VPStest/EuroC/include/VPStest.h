#pragma once

#include<opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>

#include "DataBase.h"
#include "Converter.h"
#include "Parameter.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "utils.h"
#include "BoostArchiver.h"

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

extern cv::Mat K;
struct QueryDB;
class VPStest
{

// DataBase

public:
    
    VPStest()
    {
        ORBfeatureAndDescriptor(nFeatures, scaleFactor, nlevels, iniThFAST, minThFAST);
    }


    cv::Ptr<cv::ORB> orb = cv::ORB::create(featureNum);
    std::map<int, std::vector<cv::Point3f>> MatchDB3dPoints;
    std::map<int, std::vector<cv::Point2f>> MatchQ2dPoints;
    std::vector<int> CandidateKFid;

    void SetCandidateKFid(DBoW2::QueryResults ret);

    int FindReferenceKF(DataBase* DB, cv::Mat QDescriptor, std::vector<cv::KeyPoint> QKeypoints, cv::Mat Qimg);
    double PnPInlierRatio(int KFid);
    double VPStestToReferenceKF(DataBase* DB, cv::Mat QDescriptor, std::vector<cv::KeyPoint> QKeypoints, int KFid, Eigen::Matrix4d &Pose, cv::Mat &Inliers, std::vector<cv::DMatch> &GoodMatches_);

    int LoadDBfile(std::string dbFilePath, DataBase *DB);
    void InputDBdescriptorTovoc(DataBase *DB, OrbDatabase *db);
    int Loadgt(std::string queryGtTrajectoryPath);

    void InputQuerydb(QueryDB *query, std::string timeStampfilePath, std::string queryImgdirPath);

    cv::Mat InputQueryImg(std::string QueryFile);
    cv::Mat InputQueryImg(const QueryDB query, int imageNum);
    std::vector<cv::KeyPoint> ORBFeatureExtract(cv::Mat img);
    cv::Mat ORBDescriptor(cv::Mat img, std::vector<cv::KeyPoint> keypoints);
    std::vector<cv::DMatch> ORBDescriptorMatch(cv::Mat trainDescriptor, cv::Mat queryDescriptor);
    int FindKFImageNum(int KFid, DataBase* DB, std::vector<double> timestamps);
    void InlierMatchResult(std::vector<cv::DMatch> &Matches, cv::Mat Inliers);

public:
    
    int featureNum;
    int nFeatures = 4000;
    float scaleFactor = 1.2;
    int nlevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;
    ORBextractor ORBfeatureAndDescriptor;
    
    // result
    Vector6d QueryPose;


    // gt pose
    std::vector<Vector6d> gtPoses;


};

struct QueryDB
{
    // Query 
    std::vector<cv::Mat> qImgs;
    std::vector<double> qTimestamps;
    cv::Mat qMask;
    cv::Mat qDescriptor;
    std::vector<cv::KeyPoint> qKeypoints;

    void clear()
    {
        qMask.release();
        qDescriptor.release();
        qKeypoints.clear();
    }
        
};

