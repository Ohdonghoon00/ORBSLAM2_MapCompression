#include "BoostArchiver.h"
#include "DataBase.h"
#include "VPStest.h"
#include "ORBVocabulary.h"
#include "Parameter.h"
#include "Converter.h"
#include "ORBextractor.h"
#include "map_viewer.h"
#include "VPStestResult.h"

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
        timestamps.push_back(std::stod(line)/1e9);
    }
    
    s.close();
  
    // Save PnPinlier result
    ofstream file;
    file.open("MH01_MH02_VPStest_35%_PnP_Result.txt");

    // Save trajectory result
    ofstream traj_file;
    traj_file.open("MH01_MH02_VPStest_35%_Pose_result.txt");

    // test file result
    ofstream test_file;
    test_file.open("MH01_MH02_test_35%_Result.txt");

    std::string filepath = "MH01_MH02_VPStest_35%_result.bin";

    time_t start = time(NULL);
    
    ///////// VPS TEST //////////
    while(image_num < Queryimgs.size())
    {
        // glClear(GL_COLOR_BUFFER_BIT);

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
        std::cout << "Find Reference Keyframe !! " << std::endl;
        VPStest.SetCandidateKFid(ret);

        int ReferenceKFId = VPStest.FindReferenceKF(DB, QDescriptors, QKeypoints);
        std::cout << " Selected Keyframe num : " << ReferenceKFId << std::endl;

        // VPS test to ReferenceKF
        Eigen::Matrix4d Pose;
        cv::Mat Inliers;
        std::vector<cv::DMatch> Matches;
        // std::vector<float> ReprojectionErr;
        double PnPInlierRatio = VPStest.VPStestToReferenceKF(DB, QDescriptors, QKeypoints, ReferenceKFId, Pose, Inliers, Matches);
        std::cout << " PnPInlier Ratio of Selected Keyframe : " << PnPInlierRatio << std::endl;
        Eigen::Quaterniond q = ToQuaternion(Pose);
        // Vector6d poseresult = To6DOF(Pose);


        

        traj_file <<    timestamps[image_num] << " " << Pose(0, 3) << " " << Pose(1, 3) << " " << Pose(2, 3) << 
                " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        
        file << timestamps[image_num] <<  " " << PnPInlierRatio << " " << Inliers.rows << std::endl;

        test_file << timestamps[image_num] << " " << image_num << " " << ReferenceKFId << " " << DBow2HighScoreKFId << std::endl;
            
        // Print Result
        std::cout << "Place Recognition Result !!!!!!!!!!!! " << std::endl;
        std::cout << "Query image Num : " << image_num << std::endl;
        std::cout << "DataBase image Num  : " <<  ReferenceKFId << std::endl;
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
        std::cout << " Match size : " << Matches.size() << std::endl;

        cv::drawMatches(QueryImg, QKeypoints, DB->LeftKFimg[ReferenceKFId], DB2dMatchForDraw, Matches, MatchImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("MatchImg", MatchImg);

        // VPS Result Pose
        show_trajectory_keyframe(Pose, 0.0, 0.0, 1.0, 0.5, 3.5);
        glFlush();
        
        // Save Result
        // SaveVPStestResult->MatchingImg[image_num] = MatchImg;
        // // SaveVPStestResult->PnPpose[image_num] = poseresult;
        // SaveVPStestResult->PnPInlierRatio[image_num] = PnPInlierRatio;
        // SaveVPStestResult->PnPInliers[image_num] = Inliers.rows;
        // SaveVPStestResult->DBoW2ResultImgNum[image_num] = ReferenceKFId;



        // cv::waitKey();
        
        image_num++;
        std::cout << std::endl;

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
    file << " Total VPS time is  : " << duration << " sec" << std::endl;
    std::cout << " Total VPS time is  : " << duration << " sec" << std::endl;
    
    traj_file.close();
    file.close();
    test_file.close();
    
    cv::waitKey();

    return 0;
}

            

    

     


