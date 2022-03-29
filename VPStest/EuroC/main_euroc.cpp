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
    
    

    VPStest VPStest;
    QueryDB query;
    
    
    // VPStestResult* SaveVPStestResult;
    // SaveVPStestResult = new VPStestResult();
    DataBase* DB;
    double duration(0.0), duration_(0.0);
    int image_num = 0;
    int failNum = 0;

    // Load Map
    std::cout << "Load DB ... " << std::endl;
    std::string dataBasefilePath = argv[1];
    VPStest.LoadDBfile(dataBasefilePath, DB);
    std::cout << "KF num : " << DB->KFtoMPIdx.size() << std::endl;
    std::cout << "landMarks num : " << DB->Landmarks.size() << std::endl;

    
    // Load voc
    std::cout << "Load voc ... " << std::endl;
    ORBVocabulary voc;
    voc.loadFromTextFile(argv[2]);
    
    std::cout << "Copy db descriptor to voc ... " << std::endl;
    OrbDatabase db(voc, false, 0); // false = do not use direct index
    VPStest.InputDBdescriptorTovoc(DB, &db);
    std::cout << db << std::endl;
    
    // Load timestamp and query imgs
    std::string timestampPath = argv[3];
    std::string queryImgdirPath = argv[4];
    
    // Load gt trajectory for Evaluation
    std::string queryGtTrajectoryPath = argv[5];
    VPStest.Loadgt(queryGtTrajectoryPath);

        

    


    // Save PnPinlier result
    ofstream TimeStampFile;
    TimeStampFile.open("MH01_MH02_VPStest_original_TimeStamp(full_ba).txt");

    // Save trajectory result
    ofstream EstimateTraj;
    EstimateTraj.open("MH01_MH02_VPStest_original_Pose(full_ba).txt");

    // PnP and DBoW2 Result
    ofstream ResultFile;
    ResultFile.open("MH01_MH02_original_Result(full_ba).txt");

    // std::string filepath = "MH01_MH02_VPStest_original_result.bin";

    time_t start = time(NULL);
    
    ///////// VPS TEST //////////
    while(image_num < Queryimgs.size())
    {
        // glClear(GL_COLOR_BUFFER_BIT);
        query.clear();
        cv::Mat QueryImg = VPStest.InputQueryImg(query, image_num);
        std::cout << " Image Num is  :  " << image_num << "      !!!!!!!!!!!!!!!!!!!!" << std::endl;

        // debug
        if(image_num < 338){
            image_num++;
            continue;
        }

        // Extract query feature and descriptor
        // ORBfeatureAndDescriptor(QueryImg, query.mask, query.qKeypoints, query.qDescriptors);
    
        // // Place Recognition
        // QueryResults ret;
        // ret.clear();
    
        // std::vector<cv::Mat> VQDescriptors = MatToVectorMat(QDescriptors);
        // db.query(VQDescriptors, ret, 20);
        // std::cout << ret << std::endl;
        // std::cout << "High score keyframe  num : "  << ret[0].Id << std::endl;
        // // "       word num : " << ret[0].nWords << std::endl;

        // int DBow2HighScoreKFId = ret[0].Id;

        // // FindReferenceKF
        // std::cout << "Find Reference Keyframe !! " << std::endl;
        // VPStest.SetCandidateKFid(ret);

        // int ReferenceKFId = VPStest.FindReferenceKF(DB, QDescriptors, QKeypoints, QueryImg);
        // std::cout << " Selected Keyframe num : " << ReferenceKFId << std::endl;
        // if(ReferenceKFId == -1){
        //     image_num++;
        //     failNum++;
        //     // cv::imshow("query", QueryImg);
        //     // cv::imshow("db top1", DB->LeftKFimg[DBow2HighScoreKFId]);
        //     // cv::waitKey();
        //     continue;
        // }    


        // // VPS test to ReferenceKF
        // Eigen::Matrix4d Pose;
        // cv::Mat Inliers;
        // std::vector<cv::DMatch> Matches;
        // // std::vector<float> ReprojectionErr;
        // double PnPInlierRatio = VPStest.VPStestToReferenceKF(DB, QDescriptors, QKeypoints, ReferenceKFId, Pose, Inliers, Matches);
        // std::cout << " PnPInlier Ratio of Selected Keyframe : " << PnPInlierRatio << std::endl;
        // Vector6d PnPpose = To6DOF(Pose);




        // // RSE Error (root - square error)
        // Eigen::Matrix4d RelativePose = To44RT(QueryGTTrajectory[image_num]).inverse() * To44RT(PnPpose);
        
        // // trans
        // Eigen::Vector3d RelativeTrans;
        // RelativeTrans << RelativePose(0, 3), RelativePose(1, 3), RelativePose(2, 3);
        // double Error_Trans = std::sqrt(RelativeTrans.dot(RelativeTrans));
        
        // // rotation
        // Eigen::Matrix3d RelativeRot_ = RelativePose.block<3, 3>(0, 0);
        // Eigen::Vector3d RelativeRot = ToVec3(RelativeRot_);
        // double Error_Rot = std::sqrt(RelativeRot.dot(RelativeRot));


        // TimeStampFile << setprecision(19) << timestamps[image_num] << " " << image_num << std::endl;

        // EstimateTraj << PnPpose[0] << " " << PnPpose[1] << " " << PnPpose[2] << 
        //         " " << PnPpose[3] << " " << PnPpose[4] << " " << PnPpose[5] << std::endl;
        
        // ResultFile << PnPInlierRatio << " " << Inliers.rows << " " << DBow2HighScoreKFId << " " << ReferenceKFId << " " << Error_Trans << " " << Rad2Degree(Error_Rot) << std::endl;

        // // Print Result
        // std::cout << "Place Recognition Result !!!!!!!!!!!! " << std::endl;
        // std::cout << "Query image Num : " << image_num << std::endl;
        // std::cout << "DataBase image Num  : " <<  ReferenceKFId << std::endl;
        // std::cout << "SolvePnPResult  !!!!!!!!!!!! " << std::endl;
        // std::cout << "SolvePnP Estimate Pose : " << PnPpose.transpose() << std::endl;
        // std::cout << "SolvePnP GT Pose : " << QueryGTTrajectory[image_num].transpose() << std::endl;
        // std::cout << "TransError : " << Error_Trans << std::endl;
        // std::cout << "RotError : " << Rad2Degree(Error_Rot) << std::endl;
        // std::cout << " SolvePnPInlier Ratio : " << PnPInlierRatio << std::endl;
        // std::cout << " Inliers num  : " << Inliers.rows << std::endl;

        // // Draw Map point and keyframe pose
        // // Map Points
        // std::vector<cv::Point3f> LandMarks = DB->GetKF3dPoint(ReferenceKFId);
        // for(int i = 0; i < LandMarks.size(); i++){
        //     GLdouble X_map(LandMarks[i].x), Y_map(LandMarks[i].y), Z_map(LandMarks[i].z);
        //     show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 0.0, 0.01);
        // }
            
        // // For Draw Inliers Match
        // std::vector<cv::KeyPoint> DB2dMatchForDraw = DB->GetKF2dPoint(ReferenceKFId);
        // cv::Mat allMatchImg;
        // std::cout << " Total Match size : " << Matches.size() << std::endl;
        // std::vector<cv::DMatch> goodMatches(Matches.begin(), Matches.begin() + 100);
        // cv::drawMatches(QueryImg, QKeypoints, DB->LeftKFimg[ReferenceKFId], DB2dMatchForDraw, Matches, allMatchImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        // cv::imshow("allMatchImg", allMatchImg);        
        
        // VPStest.InlierMatchResult(Matches, Inliers);
        // cv::Mat InlierMatchImg;
        // std::cout << " Inlier Match size : " << Matches.size() << std::endl;

        // cv::drawMatches(QueryImg, QKeypoints, DB->LeftKFimg[ReferenceKFId], DB2dMatchForDraw, Matches, InlierMatchImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        // cv::imshow("inlierMatchImg", InlierMatchImg);
        // // cv::imshow("Img", QueryImg);
        // // VPS Result Pose
        // show_trajectory_keyframe(Pose, 0.0, 0.0, 1.0, 0.1, 0.2);
        // glFlush();
        
        // // Save Result
        // // SaveVPStestResult->MatchingImg[image_num] = MatchImg;
        // // SaveVPStestResult->PnPpose[image_num] = poseresult;
        // // SaveVPStestResult->PnPInlierRatio[image_num] = PnPInlierRatio;
        // // SaveVPStestResult->PnPInliers[image_num] = Inliers.rows;
        // // SaveVPStestResult->DBoW2ResultImgNum[image_num] = ReferenceKFId;


        // // int key = cv::waitKey(1);
        // // if(key == 32){
        // //     key = cv::waitKey();
        // // }
        // cv::waitKey();
        
        // image_num++;
        // std::cout << std::endl;

    }
    
    // std::ofstream out(filepath, std::ios_base::binary);
    // if (!out)
    // {
    //     std::cout << "Cannot Write to Database File: " << std::endl;
    //     exit(-1);
    // }
    // boost::archive::binary_oarchive oa(out, boost::archive::no_header);
    // oa << SaveVPStestResult;
    // out.close();

    time_t finish = time(NULL);
    duration = (double)(finish - start);
    
    std::cout << " Finish Visual Localization " << std::endl;
    std::cout << "Total Landmark Num : " << DB->Landmarks.size() << std::endl;
    // file << " Total VPS time is  : " << duration << " sec" << std::endl;
    std::cout << " Total VPS time is  : " << duration << " sec" << std::endl;
    std::cout << "Total image Num is : " << image_num << " " << " fail image num is : " << failNum << std::endl;
    
    TimeStampFile.close();
    EstimateTraj.close();
    ResultFile.close();
    
    cv::waitKey();

    return 0;
}

            

    

     


