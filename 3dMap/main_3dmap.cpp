#include <iostream>
#include<algorithm>
#include<fstream>
#include <sstream>
#include <cmath>

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/xfeatures2d.hpp"
// #include <opencv2/nonfree/nonfree.hpp>
// #include <opencv2/nonfree/features2d.hpp>

#include <Eigen/Dense>

#include "BoostArchiver.h"
#include "DataBase.h"
#include "Parameter.h"
#include "Converter.h"
#include "ORBextractor.h"
#include "map_viewer.h"
#include "utils.h"
#include "Map.h"
#include "Keyframe.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    //
    cv::VideoCapture video;
    std::vector<Vector6d> KFgtPoses;
    int KFcnt = 0;
    cv::Mat K = GetK(IntrinsicData);
    std::cout << K << std::endl;
    // Extract feature
    int nFeatures = 2000;
    float scaleFactor = 1.2;
    int nlevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;
    // ORBextractor ORBfeatureAndDescriptor(nFeatures, scaleFactor, nlevels, iniThFAST, minThFAST);
    // cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures);
    cv::Ptr<Feature2D> sift = cv::xfeatures2d::SIFT::create(nFeatures);

    //
    Keyframe Last_KF, Curr_KF;
    Map MapDB;
    std::map< int, std::vector<cv::DMatch>> matches;
    int GoodMatchNum = 300;

    // Load img and gt Pose
    // std::string KFimgPath_ = argv[1];
    // std::string KFimgPath = KFimgPath_ + "/%04d.png";
    // if(!video.open(KFimgPath)) return -1;

    std::string KFgtPosePath = argv[3];
    ReadgtPose(KFgtPosePath, &KFgtPoses);

    // Load timestamp
    std::string timestampPath = argv[1];
    ifstream s;
    s.open(timestampPath);
    std::string line;
    std::vector<double> timestamps;
    std::vector<cv::Mat> Queryimgs1;
    std::vector<cv::Mat> Queryimgs2;
    // Load Query Img
    std::cout << " Input Query Img " << std::endl;
    std::string QueryPath = argv[2];

    while(std::getline(s, line)){
        
        std::stringstream ss;
        ss << line;
        std::string QueryImgsPath1 = QueryPath + "/cam0/data/" + ss.str() +".png";
        cv::Mat image1 = cv::imread(QueryImgsPath1);
            std::string QueryImgsPath2 = QueryPath + "/cam1/data/" + ss.str() +".png";
        cv::Mat image2 = cv::imread(QueryImgsPath2);
        // Queryimgs1.push_back(img1);
        // Queryimgs2.push_back(img2);
        timestamps.push_back(std::stod(line) * 10e-10);
    
    
     


    // Main
    // while(true)
    // {
        // cv::Mat image1 = Queryimgs1[KFcnt];
        // cv::Mat image2 = Queryimgs2[KFcnt];
        // video >> image;
        // if(image.empty()) break;
        // if (image.channels() > 1) cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
        // MapDB.KFimg[KFcnt] = image;
        
        std::cout << " KF img Num : " << KFcnt << "  @@@@@@@@@@@@@@@@@ " << std::endl;
    
        /////////// Extract Feature and match //////////////
            
            // extract
        cv::Mat mask, Descriptors;
        std::vector<cv::Point2f> keypoint;
        Curr_KF.EraseClass();
        Last_KF.EraseClass();
        // ORBfeatureAndDescriptor(image1, Last_KF.mask, Last_KF.KeyPoints, Last_KF.Descriptors);
        // ORBfeatureAndDescriptor(image2, Curr_KF.mask, Curr_KF.KeyPoints, Curr_KF.Descriptors);
        // cv::goodFeaturesToTrack(image, Curr_KF.keypoint, nFeatures, 0.01, 10);
        // Curr_KF.KeyPoints = Converter::Point2f2KeyPoint(Curr_KF.keypoint);
        // orb->detectAndCompute(image1, Last_KF.mask, Last_KF.KeyPoints, Last_KF.Descriptors);
        // orb->detectAndCompute(image2, Curr_KF.mask, Curr_KF.KeyPoints, Curr_KF.Descriptors);
        sift->detectAndCompute(image1, Last_KF.mask, Last_KF.KeyPoints, Last_KF.Descriptors);
        sift->detectAndCompute(image2, Curr_KF.mask, Curr_KF.KeyPoints, Curr_KF.Descriptors);
        std::cout << "feature num : " << Curr_KF.KeyPoints.size() << std::endl;
        // initial value
        // if(KFcnt == 0){
        //     KFcnt++;
        //     Last_KF = Curr_KF;
        //     continue;
        // }

            // match
        // cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, true);
        // cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
        std::vector<cv::DMatch> DescriptorMatch;
        matcher->match(Last_KF.Descriptors, Curr_KF.Descriptors, DescriptorMatch);
        std::sort(DescriptorMatch.begin(), DescriptorMatch.end());
        matches[KFcnt - 1] = DescriptorMatch;
        
        // Last_KF.keypoint.clear();
        for(int i = 0; i < matches[KFcnt - 1].size(); i++){
            Last_KF.keypoint.push_back(Last_KF.KeyPoints[matches[KFcnt - 1][i].queryIdx].pt);
            Curr_KF.keypoint.push_back(Curr_KF.KeyPoints[matches[KFcnt - 1][i].trainIdx].pt);
        }
        cv::Mat E, inlier_mask, R, t;
        std::cout << Last_KF.keypoint.size() << " " << Curr_KF.keypoint.size() << std::endl;
        E = cv::findEssentialMat(Last_KF.keypoint, Curr_KF.keypoint, fx, c, cv::RANSAC, 0.99, 1, inlier_mask);
        int inlierNum = cv::recoverPose(E, Last_KF.keypoint, Curr_KF.keypoint, R, t, fx, c, inlier_mask);
        std::cout << inlierNum << std::endl;
        // Triangulation
        cv::Mat P0 = K * Vec6To34Mat(KFgtPoses[KFcnt - 1]);
        cv::Mat P1 = K * Vec6To34Mat(KFgtPoses[KFcnt]);
        cv::Mat X;
        cv::triangulatePoints(P0, P1, Last_KF.keypoint, Curr_KF.keypoint, X);
        std::vector<cv::Point3f> MapPts = ToXYZ(X);
        std::vector<float> ReprojErr = ReprojectionError(MapPts, Last_KF.keypoint, To44RT(KFgtPoses[KFcnt - 1]));
        /////////// Remove Outlier ////////////////////
            
            // hamming distance
        std::vector<cv::DMatch> GoodDescriptorMatch(DescriptorMatch.begin(), DescriptorMatch.begin() + GoodMatchNum);
        
        // Draw Match Img
        cv::Mat MatchImg;
        cv::drawMatches(image1, Last_KF.KeyPoints, image2, Curr_KF.KeyPoints, GoodDescriptorMatch, MatchImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DEFAULT);
        cv::imshow("Matchimg", MatchImg);

        

        cv::waitKey();
        // Last_KF.EraseClass();
        // Last_KF = Curr_KF;
        KFcnt++;
    // }
}
s.close();
        // descriptor distance and reprojection err?

    // id correspondance




    // Full BA

    return 0;
}