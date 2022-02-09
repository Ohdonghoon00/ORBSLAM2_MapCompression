#include "BoostArchiver.h"
#include "DataBase.h"
#include "VPStest.h"
#include "ORBVocabulary.h"
#include "Parameter.h"
#include "Converter.h"
#include "ORBextractor.h"
#include "map_viewer.h"
#include "VPStestResult.h"
#include "Evaluatation.h"

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


using namespace std;
using namespace cv;
using namespace DBoW2;

int main(int argc, char **argv)
{
    int nFeatures = 4000;
    float scaleFactor = 1.2;
    int nlevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;
    VPStest VPStest(4000);
    ORBextractor ORBfeatureAndDescriptor(nFeatures, scaleFactor, nlevels, iniThFAST, minThFAST);
    // ORBextractor ORBfeatureAndDescriptor(4000, scaleFactor, nlevels, iniThFAST, minThFAST);
    std::vector<Eigen::VectorXf> LcamGTPose, RcamGTPose;
    DataBase* DB;
    int KF_Img_Num = 0;

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
    // ORBVocabulary voc(argv[2]);
    ORBVocabulary voc;
    voc.loadFromTextFile(argv[2]);
    std::cout << "copy voc to db" << std::endl;
    OrbDatabase db(voc, false, 0); // false = do not use direct index
    std::string DataPath = argv[3];
    
    // Load timestamp
    std::string timestampPath = DataPath + "/times.txt";
    ifstream s;
    s.open(timestampPath);
    std::string line;
    std::vector<double> timestamps;

    while(std::getline(s, line))
        timestamps.push_back(std::stod(line));
    
    s.close();
    

    std::vector<cv::Mat> DB_images;
    // Input DB descriptor to voc
    std::cout << " KF img num : " << DB->KFtoMPIdx.size() << std::endl;
    for(size_t i = 0; i < DB->KFtoMPIdx.size(); i++){
        std::vector<cv::Mat> KFDescriptor;
        KFDescriptor.clear();
        KFDescriptor = MatToVectorMat(DB->GetKFMatDescriptor(i));
        std::cout << " KF num : " << i << "     Keypoint num : " << (DB->GetKFMatDescriptor(i)).size() <<  "    " << DB->KFtoMPIdx[i].size() << std::endl;  
        cv::Mat DB_image = DB->LeftKFimg[i];
        // int KF_num = VPStest.FindKFImageNum(i, DB, timestamps);
        // std::stringstream DBimagePath;  
        // DBimagePath << DataPath + "/image_0/" << std::setfill('0') << std::setw(6) << KF_num << ".png";
        // cv::Mat DB_image = cv::imread(DBimagePath.str(), cv::ImreadModes::IMREAD_GRAYSCALE);
        cv::Mat mask_, QDescriptors_;
        std::vector<cv::KeyPoint> QKeypoints_;
        ORBfeatureAndDescriptor(DB_image, mask_, QKeypoints_, QDescriptors_);        
        std::vector<cv::Mat> DBDescriptors = MatToVectorMat(QDescriptors_);
        db.add(DBDescriptors);
    }
    std::cout << db << std::endl;


    
    // Load GT Pose 
    ifstream s_;
    s_.open(argv[4]);
    std::string line_;

    while(std::getline(s_, line_)){

        
        std::string value;
        std::vector<std::string> values;
        std::stringstream ss(line_);
        
        while(std::getline(ss, value, ' '))
            values.push_back(value);
        Eigen::VectorXf Leftcam(7);
        Leftcam <<    std::stof(values[1]), std::stof(values[2]), std::stof(values[3]), 
                std::stof(values[4]), std::stof(values[5]), std::stof(values[6]), std::stof(values[7]);
        Eigen::VectorXf Rightcam(7);
        Rightcam = LeftCamToRightCam(Leftcam);
        RcamGTPose.push_back(Rightcam);
        LcamGTPose.push_back(Leftcam);
    }
    s_.close();

    int image_num = VPStest.FindKFImageNum(std::stoi(argv[5]), DB, timestamps);
    std::cout << image_num << std::endl;

    // Input Query
    std::cout << " Input Query Img " << std::endl;
    std::stringstream jkk;
    jkk << DataPath << "/image_1/" << std::setfill('0') << 
                                std::setw(6) << image_num << ".png";
    std::string QueryPath = jkk.str(); 
    // = DataPath + "/image_1/" << std::setfill('0') << 
    //                             std::setw(5) << argv[4] + ".png";
    cv::VideoCapture video;
    if(!video.open(QueryPath)){
        std::cout << " No query image " << std::endl;
        return -1;
    }

    cv::Mat QueryImg;
    video >> QueryImg;
    if (QueryImg.channels() > 1) cv::cvtColor(QueryImg, QueryImg, cv::COLOR_RGB2GRAY);

    cv::Mat mask, QDescriptors;
    std::vector<cv::KeyPoint> QKeypoints;
    ORBfeatureAndDescriptor(QueryImg, mask, QKeypoints, QDescriptors);

    // Place Recognition
    QueryResults ret;
    // ret.clear();

    std::vector<cv::Mat> VQDescriptors = MatToVectorMat(QDescriptors);
    db.query(VQDescriptors, ret, 20);
    std::cout << ret << std::endl;
    std::cout << "High score keyframe  num : "  << ret[0].Id << std::endl;

    // FindReferenceKF
    std::cout << "Find Reference Keyframe !! " << std::endl;
    VPStest.SetCandidateKFid(ret);
    int ReferenceKFId = VPStest.FindReferenceKF(DB, QDescriptors, QKeypoints);
    // int ReferenceKFId = ret[0].Id;
    std::cout << " Selected Keyframe num : " << ReferenceKFId << std::endl;
            
    int DBoW2Result_KF_imageNum = VPStest.FindKFImageNum(ret[0].Id, DB, timestamps);
    std::cout << "DBoW2Result KF image Num :  " << DBoW2Result_KF_imageNum << std::endl;        
            
    // For Draw image 
    int Selected_KF_imageNum = VPStest.FindKFImageNum(ReferenceKFId, DB, timestamps);
    std::cout << "Selecte KF image Num :  " << Selected_KF_imageNum << std::endl;

    // VPS test to ReferenceKF
    Eigen::Matrix4d Pose;
    cv::Mat Inliers;
    std::vector<cv::DMatch> Matches;
    // std::vector<float> ReprojectionErr;
    double PnPInlierRatio = VPStest.VPStestToReferenceKF(DB, QDescriptors, QKeypoints, ReferenceKFId, Pose, Inliers, Matches);
    std::cout << " PnPInlier Ratio of Selected Keyframe : " << PnPInlierRatio << std::endl;
    Eigen::Quaterniond q = ToQuaternion(Pose);
    
            // Print Result
            std::cout << "Place Recognition Result !!!!!!!!!!!! " << std::endl;
            std::cout << "Query image Num : " << image_num << std::endl;
            std::cout << "DataBase image Num  : " <<  Selected_KF_imageNum << std::endl;
            std::cout << "SolvePnPResult  !!!!!!!!!!!! " << std::endl;
            std::cout << Pose << std::endl;
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
    std::vector<cv::KeyPoint> DB2dMatchForDraw = DB->GetKF2dPoint(ReferenceKFId);
    VPStest.InlierMatchResult(Matches, Inliers);
    cv::Mat MatchImg;
    std::cout << Matches.size() << std::endl;

    cv::Mat imagese;
    cv::drawKeypoints(DB->LeftKFimg[ReferenceKFId], DB2dMatchForDraw, imagese);
    cv::imshow("aa", imagese);

    cv::drawMatches(QueryImg, QKeypoints, DB->LeftKFimg[ReferenceKFId], DB2dMatchForDraw, Matches, MatchImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // cv::Size a = MatchImg.size();
    cv::resize(MatchImg, MatchImg, cv::Size(1737, 376));
    cv::imshow("MatchImg", MatchImg);

    // Evaluation Pose
    TranslationError(LcamGTPose, Pose, ReferenceKFId);


    cv::waitKey();            
    // VPS Result Pose
    show_trajectory_keyframe(Pose, 0.0, 0.0, 1.0, 0.5, 3.5);
    glFlush();

    KF_Img_Num++;
    return 0;
}