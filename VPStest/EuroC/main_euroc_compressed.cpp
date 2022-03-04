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
    VPStest VPStest(4000);
    ORBextractor ORBfeatureAndDescriptor(nFeatures, scaleFactor, nlevels, iniThFAST, minThFAST);
    
    VPStestResult* SaveVPStestResult;
    SaveVPStestResult = new VPStestResult();
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


    // for(int i = 0; DB->LeftKFimg.size(); i++){
    //     cv::imshow("KFimg", DB->LeftKFimg[i]);
    //     std::cout << setprecision(19) << DB->timestamps[i] << std::endl;
    //     cv::waitKey();
    // }
    
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
        std::vector<cv::Mat> DBDescriptors = MatToVectorMat(DBDescriptor_);
        db.add(DBDescriptors);
        std::cout << setprecision(19) << DB->timestamps[i] << std::endl;
    }
    std::cout << db << std::endl;
        
    // Load timestamp
    std::string timestampPath = argv[3];
    ifstream s;
    s.open(timestampPath);
    std::string line;
    std::vector<double> timestamps;
    std::vector<cv::Mat> Queryimgs;
    
    // Load Query Img
    std::cout << " Input Query Img " << std::endl;
    std::string QueryPath = argv[4];

    while(std::getline(s, line)){
        
        std::stringstream ss;
        ss << line;
        std::string QueryImgsPath = QueryPath + "/" + ss.str() +".png";
        cv::Mat img = cv::imread(QueryImgsPath);
        Queryimgs.push_back(img);
        timestamps.push_back(std::stod(line) * 10e-10);
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

    // Original Map result TimeStamp and Selected KF
    std::string OriginalResultPath = argv[6];
    std::string OriginalTimeStampPath = OriginalResultPath + "/MH01_MH02_VPStest_original_TimeStamp.txt";
    std::string OriginalResultSelPath = OriginalResultPath + "/MH01_MH02_original_Result.txt";
    
    ifstream OriginalTimeStampFile(OriginalTimeStampPath, std::ifstream::in);
    ifstream OriginalResultSelFile(OriginalResultSelPath, std::ifstream::in);

    std::vector<double> OriginalResultTimeStamps;
    std::vector<int> OriginalResultSelectedKF;

    std::string line1;
    while(std::getline(OriginalTimeStampFile, line1)){
        std::string value;
        std::vector<std::string> values;

        std::stringstream ss(line1);
        while(std::getline(ss, value, ' '))
            values.push_back(value);
        OriginalResultTimeStamps.push_back(std::stod(values[0]));
    }

    std::string line2;
    while(std::getline(OriginalResultSelFile, line2)){
        std::string value;
        std::vector<std::string> values;

        std::stringstream ss(line2);
        while(std::getline(ss, value, ' '))
            values.push_back(value);
        OriginalResultSelectedKF.push_back(std::stoi(values[3]));
    }

    // Save PnPinlier result
    ofstream TimeStampFile;
    TimeStampFile.open("MH01_MH02_VPStest_20%_TimeStamp.txt");

    // Save trajectory result
    ofstream EstimateTraj;
    EstimateTraj.open("MH01_MH02_VPStest_20%_Pose.txt");

    // PnP and DBoW2 Result
    ofstream ResultFile;
    ResultFile.open("MH01_MH02_20%_Result.txt");

    std::string filepath = "MH01_MH02_VPStest_20%_result.bin";

    time_t start = time(NULL);
    
    ///////// VPS TEST //////////
    while(image_num < Queryimgs.size())
    {
        // glClear(GL_COLOR_BUFFER_BIT);
        // std::cout << setprecision(19) << timestamps[image_num] << std::endl;
        // std::cout << setprecision(19) << OriginalResultTimeStamps[image_num] << std::endl;
        auto it = std::find(OriginalResultTimeStamps.begin(), OriginalResultTimeStamps.end(), timestamps[image_num]);
        if(it == OriginalResultTimeStamps.end()){
            image_num++;
            // std::cout << "a" << std::endl;
            continue;
        }
        int index = it - OriginalResultTimeStamps.begin();

        
        cv::Mat QueryImg = Queryimgs[image_num];
        if(QueryImg.empty()) {
            std::cout << " Error at input query img " << std::endl; 
            break;
        }
        if (QueryImg.channels() > 1) cv::cvtColor(QueryImg, QueryImg, cv::COLOR_RGB2GRAY);
        std::cout << " Image Num is  :  " << image_num << "      !!!!!!!!!!!!!!!!!!!!" << std::endl;

        cv::Mat mask, QDescriptors;
        std::vector<cv::KeyPoint> QKeypoints;
        ORBfeatureAndDescriptor(QueryImg, mask, QKeypoints, QDescriptors);
    
        // Place Recognition
        QueryResults ret;
        ret.clear();
    
        std::vector<cv::Mat> VQDescriptors = MatToVectorMat(QDescriptors);
        db.query(VQDescriptors, ret, 10);
        std::cout << ret << std::endl;
        std::cout << "High score keyframe  num : "  << ret[0].Id << std::endl;
        // "       word num : " << ret[0].nWords << std::endl;

        int DBow2HighScoreKFId = ret[0].Id;

        // FindReferenceKF
        // std::cout << "Find Reference Keyframe !! " << std::endl;
        // VPStest.SetCandidateKFid(ret);

        int ReferenceKFId = OriginalResultSelectedKF[index];
        std::cout << "Reference Keyframe : " << ReferenceKFId << std::endl;
        // std::cout << " Selected Keyframe num : " << ReferenceKFId << std::endl;
        // if(ReferenceKFId == -1){
        //     image_num++;
        //     continue;
        // }    


        // VPS test to ReferenceKF
        Eigen::Matrix4d Pose;
        cv::Mat Inliers;
        std::vector<cv::DMatch> Matches;
        // std::vector<float> ReprojectionErr;
        double PnPInlierRatio = VPStest.VPStestToReferenceKF(DB, QDescriptors, QKeypoints, ReferenceKFId, Pose, Inliers, Matches);
        std::cout << " PnPInlier Ratio of Selected Keyframe : " << PnPInlierRatio << std::endl;
        Vector6d PnPpose = To6DOF(Pose);


        // draw Total MatchImg
        cv::Mat TotalMatchImg;
        std::cout << " Match size : " << Matches.size() << std::endl;
        std::vector<cv::KeyPoint> DB2dMatchForDraw = DB->GetKF2dPoint(ReferenceKFId);
        cv::drawMatches(QueryImg, QKeypoints, DB->LeftKFimg[ReferenceKFId], DB2dMatchForDraw, Matches, TotalMatchImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("TotalMatchImg", TotalMatchImg);


        // RSE Error (root - square error)
        Eigen::Matrix4d RelativePose = To44RT(QueryGTTrajectory[image_num]).inverse() * To44RT(PnPpose);
        
        // trans
        Eigen::Vector3d RelativeTrans;
        RelativeTrans << RelativePose(0, 3), RelativePose(1, 3), RelativePose(2, 3);
        double Error_Trans = std::sqrt(RelativeTrans.dot(RelativeTrans));
        
        // rotation
        Eigen::Matrix3d RelativeRot_ = RelativePose.block<3, 3>(0, 0);
        Eigen::Vector3d RelativeRot = ToVec3(RelativeRot_);
        double Error_Rot = std::sqrt(RelativeRot.dot(RelativeRot));


        TimeStampFile << setprecision(19) << timestamps[image_num] << " " << image_num << std::endl;

        EstimateTraj << PnPpose[0] << " " << PnPpose[1] << " " << PnPpose[2] << 
                " " << PnPpose[3] << " " << PnPpose[4] << " " << PnPpose[5] << std::endl;
        
        ResultFile << PnPInlierRatio << " " << Inliers.rows << " " << DBow2HighScoreKFId << " " << ReferenceKFId << " " << Error_Trans << " " << Rad2Degree(Error_Rot) << std::endl;

        // Print Result
        std::cout << "Place Recognition Result !!!!!!!!!!!! " << std::endl;
        std::cout << "Query image Num : " << image_num << std::endl;
        std::cout << "DataBase image Num  : " <<  ReferenceKFId << std::endl;
        std::cout << "SolvePnPResult  !!!!!!!!!!!! " << std::endl;
        std::cout << "SolvePnP Estimate Pose : " << PnPpose.transpose() << std::endl;
        std::cout << "SolvePnP GT Pose : " << QueryGTTrajectory[image_num].transpose() << std::endl;
        std::cout << "TransError : " << Error_Trans << std::endl;
        std::cout << "RotError : " << Rad2Degree(Error_Rot) << std::endl;
        std::cout << " SolvePnPInlier Ratio : " << PnPInlierRatio << std::endl;
        std::cout << " Inliers num  : " << Inliers.rows << std::endl;
        
        // debug
        for(int i = 0; i < DB2dMatchForDraw.size(); i++){
            std::cout << DB2dMatchForDraw[i].pt << " ";
        }
        std::cout << std::endl;
        // Draw Map point and keyframe pose
        // Map Points
        std::vector<cv::Point3f> LandMarks = DB->GetKF3dPoint(ReferenceKFId);
        for(int i = 0; i < LandMarks.size(); i++){
            GLdouble X_map(LandMarks[i].x), Y_map(LandMarks[i].y), Z_map(LandMarks[i].z);
            show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 0.0, 0.01);
        }
            
        // For Draw Inliers Match
        
        VPStest.InlierMatchResult(Matches, Inliers);
        cv::Mat InlierMatchImg;
        std::cout << " Match size : " << Matches.size() << std::endl;

        cv::drawMatches(QueryImg, QKeypoints, DB->LeftKFimg[ReferenceKFId], DB2dMatchForDraw, Matches, InlierMatchImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("InlierMatchImg", InlierMatchImg);

        // VPS Result Pose
        show_trajectory_keyframe(Pose, 0.0, 0.0, 1.0, 0.5, 3.5);
        glFlush();
        
        // Save Result
        SaveVPStestResult->InlierMatchingImg[image_num] = InlierMatchImg;
        SaveVPStestResult->TotalMatchingImg[image_num] = TotalMatchImg;
        // SaveVPStestResult->PnPpose[image_num] = poseresult;
        // SaveVPStestResult->PnPInlierRatio[image_num] = PnPInlierRatio;
        // SaveVPStestResult->PnPInliers[image_num] = Inliers.rows;
        // SaveVPStestResult->DBoW2ResultImgNum[image_num] = ReferenceKFId;



        cv::waitKey();
        
        image_num++;
        std::cout << std::endl;

    }
    
    std::ofstream out(filepath, std::ios_base::binary);
    if (!out)
    {
        std::cout << "Cannot Write to Database File: " << std::endl;
        exit(-1);
    }
    boost::archive::binary_oarchive oa(out, boost::archive::no_header);
    oa << SaveVPStestResult;
    out.close();

    time_t finish = time(NULL);
    duration = (double)(finish - start);
    
    std::cout << " Finish Visual Localization " << std::endl;
    std::cout << "Total Landmark Num : " << DB->Landmarks.size() << std::endl;
    // file << " Total VPS time is  : " << duration << " sec" << std::endl;
    std::cout << " Total VPS time is  : " << duration << " sec" << std::endl;
    
    TimeStampFile.close();
    EstimateTraj.close();
    ResultFile.close();
    
    cv::waitKey();

    return 0;
}

            

    

     


