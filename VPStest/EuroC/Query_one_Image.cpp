#include "BoostArchiver.h"
#include "DataBase.h"
#include "VPStest.h"
#include "ORBVocabulary.h"
#include "Parameter.h"
#include "Converter.h"
#include "ORBextractor.h"
#include "map_viewer.h"
#include "VPStestResult.h"
#include "utils.h"

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>


#include <ctime>
#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <cstdio>
// #include <pangolin/pangolin.h>

using namespace std;
using namespace cv;
using namespace DBoW2;

int main(int argc, char **argv)
{
    
    // Viewer
    glutInit(&argc, argv);
    initialize_window();
    
    
    int nFeatures = 3000;
    float scaleFactor = 1.2;
    int nlevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;
    VPStest VPStest;
    ORBextractor ORBfeatureAndDescriptor(nFeatures, scaleFactor, nlevels, iniThFAST, minThFAST);
    
    // VPStestResult* SaveVPStestResult;
    // SaveVPStestResult = new VPStestResult();
    DataBase* DB;
    double duration(0.0), duration_(0.0);
    int image_num = 0;


    // Load Map
    std::ifstream in(argv[1], std::ios_base::binary);
    if (!in){
        std::cout << "Cannot DataBase bin file is empty!" << std::endl;
        return false;
    }
    boost::archive::binary_iarchive ia(in, boost::archive::no_header);
    ia >> DB;
    in.close();
    
    // Load voc
    std::cout << "load voc" << std::endl;
    
    ORBVocabulary voc;
    voc.loadFromTextFile(argv[2]);
    std::cout << "copy voc to db" << std::endl;
    OrbDatabase db(voc, false, 0); // false = do not use direct index
    std::cout << "KF num : " << DB->KFtoMPIdx.size() << std::endl;
    
    // Input DB descriptor to voc
    for(size_t i = 0; i < DB->KFtoMPIdx.size(); i++){

        std::cout << " KF num : " << i << "     Keypoint num : " << (DB->GetKFMatDescriptor(i)).size() << std::endl;  
        
        // Input Voc
        cv::Mat DB_image = DB->LeftKFimg[i];
        cv::Mat mask_, DBDescriptor_;
        std::vector<cv::KeyPoint> DBKeypoints_;
        ORBfeatureAndDescriptor(DB_image, mask_, DBKeypoints_, DBDescriptor_);        
        std::vector<cv::Mat> DBDescriptors = Converter::MatToVectorMat(DBDescriptor_);
        db.add(DBDescriptors);
    }
    std::cout << db << std::endl;
        
    // Load timestamp
    std::string timestampPath = argv[3];
    ifstream s;
    s.open(timestampPath);
    std::string line;
    std::vector<double> timestamps;
        
    
    std::cout << " Input Query Img " << std::endl;

    while(std::getline(s, line)){
        timestamps.push_back(std::stod(line));
    }
    
    s.close(); 
    
    // Evaluation GT Trajectory
    std::string QueryGTTrajectoryPath = argv[5];
    ifstream QueryGTTrajectoryFile(QueryGTTrajectoryPath, std::ifstream::in);

    if(!QueryGTTrajectoryFile.is_open()){
        std::cout << " GT Query Trajectory file failed to open " << std::endl;
        return EXIT_FAILURE;
    }      

    std::vector<Vector6d> QueryGTTrajectory;
    std::string QueryGTTrajectoryLine;
    while(std::getline(QueryGTTrajectoryFile, QueryGTTrajectoryLine)){
        
        std::string QueryGTTrajectoryvalue;
        std::vector<std::string> QueryGTTrajectoryvalues;

        std::stringstream ss(QueryGTTrajectoryLine);
        while(std::getline(ss, QueryGTTrajectoryvalue, ' '))
            QueryGTTrajectoryvalues.push_back(QueryGTTrajectoryvalue);

        Vector6d pose;
        pose << std::stod(QueryGTTrajectoryvalues[1]), std::stod(QueryGTTrajectoryvalues[2]), std::stod(QueryGTTrajectoryvalues[3]),
                std::stod(QueryGTTrajectoryvalues[4]), std::stod(QueryGTTrajectoryvalues[5]), std::stod(QueryGTTrajectoryvalues[6]);

        QueryGTTrajectory.push_back(pose);        
    }

    // Load Query Img
    std::string QueryPath = argv[4];
    int QueryImgNum = std::stoi(argv[6]);
    double QueryTimeStamp = timestamps[QueryImgNum];
    std::string QueryTimeStamp_string = std::to_string(lround(QueryTimeStamp));
    std::stringstream ss;
    std::cout << QueryTimeStamp_string << std::endl;
    ss << QueryTimeStamp_string;
    std::string QueryImgsPath = QueryPath + "/" + ss.str() +".png";

    time_t start = time(NULL);
    
    ///////// VPS TEST //////////
        // glClear(GL_COLOR_BUFFER_BIT);

        cv::Mat QueryImg = cv::imread(QueryImgsPath);
        if(QueryImg.empty()) {
            std::cout << " Error at input query img " << std::endl; 
            return 0;
        }
        if (QueryImg.channels() > 1) cv::cvtColor(QueryImg, QueryImg, cv::COLOR_RGB2GRAY);
        std::cout << " Image Num is  :  " << QueryImgNum << "      !!!!!!!!!!!!!!!!!!!!" << std::endl;

        cv::Mat mask, QDescriptors;
        std::vector<cv::KeyPoint> QKeypoints;
        ORBfeatureAndDescriptor(QueryImg, mask, QKeypoints, QDescriptors);
    
        // Place Recognition
        QueryResults ret;
        ret.clear();
    
        std::vector<cv::Mat> VQDescriptors = Converter::MatToVectorMat(QDescriptors);
        db.query(VQDescriptors, ret, 10);
        std::cout << ret << std::endl;
        std::cout << "High score keyframe  num : "  << ret[0].Id << std::endl;
        // "       word num : " << ret[0].nWords << std::endl;

        int DBow2HighScoreKFId = ret[0].Id;

        // FindReferenceKF
        std::cout << "Find Reference Keyframe !! " << std::endl;
        VPStest.SetCandidateKFid(ret);
        
        // Place Recognition debug
        cv::Mat DBoW2Top1Image = DB->LeftKFimg[VPStest.CandidateKFid[0]];
        cv::imshow("DBoW2Top1Image", DBoW2Top1Image);

        int ReferenceKFId = ret[0].Id;
        // int ReferenceKFId = VPStest.FindReferenceKF(DB, QDescriptors, QKeypoints);
        // std::cout << " Selected Keyframe num : " << ReferenceKFId << std::endl;

        // VPS test to ReferenceKF
        Eigen::Matrix4d Pose;
        cv::Mat Inliers;
        std::vector<cv::DMatch> Matches;
        // std::vector<float> ReprojectionErr;
        double PnPInlierRatio = VPStest.VPStestToReferenceKF(DB, QDescriptors, QKeypoints, ReferenceKFId, Pose, Inliers, Matches);
        std::cout << " PnPInlier Ratio of Selected Keyframe : " << PnPInlierRatio << std::endl;
        Vector6d PnPpose = To6DOF(Pose);




        // RSE Error (root - square error)
        Eigen::Matrix4d RelativePose = To44RT(QueryGTTrajectory[QueryImgNum]).inverse() * To44RT(PnPpose);
        
        // trans
        Eigen::Vector3d RelativeTrans;
        RelativeTrans << RelativePose(0, 3), RelativePose(1, 3), RelativePose(2, 3);
        double Error_Trans = std::sqrt(RelativeTrans.dot(RelativeTrans));
        
        // rotation
        Eigen::Matrix3d RelativeRot_ = RelativePose.block<3, 3>(0, 0);
        Eigen::Vector3d RelativeRot = ToVec3(RelativeRot_);
        double Error_Rot = std::sqrt(RelativeRot.dot(RelativeRot));
        
        // Print Result
        std::cout << "Place Recognition Result !!!!!!!!!!!! " << std::endl;
        std::cout << "Query image Num : " << QueryImgNum << std::endl;
        std::cout << "DataBase image Num  : " <<  ReferenceKFId << std::endl;
        std::cout << "SolvePnPResult  !!!!!!!!!!!! " << std::endl;
        std::cout << "SolvePnP Estimate Pose : " << PnPpose.transpose() << std::endl;
        std::cout << "SolvePnP GT Pose : " << QueryGTTrajectory[QueryImgNum].transpose() << std::endl;
        std::cout << "TransError(m) : " << Error_Trans << std::endl;
        std::cout << "RotError(degree) : " << Rad2Degree(Error_Rot) << std::endl;
        std::cout << " SolvePnPInlier Ratio : " << PnPInlierRatio << std::endl;
        std::cout << " Inliers num  : " << Inliers.rows << std::endl;

        // Draw Map point and keyframe pose
        // Map Points
        std::vector<cv::Point3f> LandMarks = DB->GetKF3dPoint(ReferenceKFId);
        for(int i = 0; i < LandMarks.size(); i++){
            GLdouble X_map(LandMarks[i].x), Y_map(LandMarks[i].y), Z_map(LandMarks[i].z);
            show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 0.0, 0.01);
        }
            
        // For Draw Inliers Match
        std::vector<cv::KeyPoint> DB2dMatchForDraw = DB->GetKFkeypoint(ReferenceKFId);
        VPStest.InlierMatchResult(Matches, Inliers);
        cv::Mat MatchImg;
        std::cout << " Match size : " << Matches.size() << std::endl;

        cv::drawMatches(QueryImg, QKeypoints, DB->LeftKFimg[ReferenceKFId], DB2dMatchForDraw, Matches, MatchImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("MatchImg", MatchImg);

        // VPS Result Pose
        show_trajectory_keyframe(Pose, 0.0, 0.0, 1.0, 0.5, 3.5);
        glFlush();

    time_t finish = time(NULL);
    duration = (double)(finish - start);
    
    std::cout << " Finish one image Visual Localization " << std::endl;
    std::cout << "Total Landmark Num : " << DB->Landmarks.size() << std::endl;
    // file << " Total VPS time is  : " << duration << " sec" << std::endl;
    std::cout << " VPS time is  : " << duration << " sec" << std::endl;
    
    cv::waitKey();

    return 0;
}

            

    

     


