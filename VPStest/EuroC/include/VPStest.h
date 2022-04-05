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
    {}


    cv::Ptr<cv::ORB> orb = cv::ORB::create(featureNum);
    std::map<int, std::vector<cv::Point3f>> MatchDB3dPoints;
    std::map<int, std::vector<cv::Point2f>> MatchQ2dPoints;
    std::vector<int> CandidateKFid;

    void SetCandidateKFid(DBoW2::QueryResults ret);

    int FindReferenceKF(DataBase* DB, QueryDB query);
    double PnPInlierRatio(int KFid);
    double VPStestToReferenceKF(DataBase* DB, QueryDB query, int KFid, Eigen::Matrix4d &Pose, cv::Mat &Inliers, std::vector<cv::DMatch> &GoodMatches_);

    int LoadDBfile(std::string dbFilePath, DataBase *DB);
    void InputDBdescriptorTovoc(DataBase *DB, OrbDatabase *db);
    int Loadgt(std::string queryGtTrajectoryPath);

    void InputQuerydb(QueryDB *query, std::string timeStampfilePath, std::string queryImgdirPath);

    cv::Mat InputQueryImg(std::string QueryFile);
    void InputQueryImg(QueryDB *query, int imageNum);
    std::vector<cv::KeyPoint> ORBFeatureExtract(cv::Mat img);
    cv::Mat ORBDescriptor(cv::Mat img, std::vector<cv::KeyPoint> keypoints);
    std::vector<cv::DMatch> ORBDescriptorMatch(cv::Mat trainDescriptor, cv::Mat queryDescriptor);
    int FindKFImageNum(int KFid, DataBase* DB, std::vector<double> timestamps);
    void InlierMatchResult(std::vector<cv::DMatch> &Matches, cv::Mat Inliers);
    std::vector<float> ReprojectionError(std::vector<cv::Point3f> WPts, std::vector<cv::Point2f> ImgPts, Eigen::Matrix4d Pose);
    void RMSError(Vector6d EsPose, Vector6d gtPose, double *err);
    void TrackOpticalFlow(cv::Mat previous, cv::Mat current, std::vector<cv::Point2f> &previous_pts, std::vector<cv::Point2f> &current_pts);


public:
    
    int featureNum;
    
    // result
    std::vector<Vector6d> candidatesPoses;

    Vector6d queryPose;
    Eigen::Matrix4d queryPose4d; 

    // for debug
    std::vector<cv::Point3f> totalLandmarks, inlierLandmarks;
    std::vector<cv::Point2f> qTotal2fpts, qInlier2fpts, projection2fpts;

    


    // gt pose
    int qImgNum; // for evaluation
    std::vector<Vector6d> gtPoses;


};

struct QueryDB
{
    QueryDB()
    {}
    
    // Total Query db 
    std::vector<cv::Mat> qImgs;
    std::vector<double> qTimestamps;
    
    // current db
    cv::Mat qImg, qMask, qDescriptor;
    std::vector<cv::KeyPoint> qKeypoints;

    // test opticalFlow
    std::vector<cv::Point2f> qKLTpts;

    void clear()
    {
        qImg.release();
        qMask.release();
        qDescriptor.release();
        qKeypoints.clear();
    }

        
};

