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
    
    
    int nFeatures = 4000;
    float scaleFactor = 1.2;
    int nlevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;
    VPStest VPStest(4000);
    ORBextractor ORBfeatureAndDescriptorDBOW2(4000, scaleFactor, nlevels, iniThFAST, minThFAST);
    ORBextractor ORBfeatureAndDescriptorP3P(4000, scaleFactor, nlevels, iniThFAST, minThFAST);
    
    VPStestResult* SaveVPStestResult;
    SaveVPStestResult = new VPStestResult();
    DataBase* DB;
    double duration(0.0), duration_(0.0);
    int image_num = 0;
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
        ORBfeatureAndDescriptorDBOW2(DB_image, mask_, QKeypoints_, QDescriptors_);        
        std::vector<cv::Mat> DBDescriptors = MatToVectorMat(QDescriptors_);
        db.add(DBDescriptors);
    }
    std::cout << db << std::endl;
    
    // Load Query Img
    std::cout << " Input Query Img " << std::endl;
    std::string QueryPath = DataPath + "/image_1/%06d.png";
    cv::VideoCapture video;
    if(!video.open(QueryPath)){
        std::cout << " No query image " << std::endl;
        return -1;
    }
    

    // Save PnPinlier result
    ofstream file;
    file.open("Kitti00_VPStest_20%_PnP_Result.txt");

    // Save trajectory result
    ofstream traj_file;
    traj_file.open("Kitti00_VPStest_20%_Pose_result.txt");

    // test file result
    ofstream test_file;
    test_file.open("Kitti00_test_20%_Result.txt");

    std::string filepath = "Kitti00_VPStest_20%_result.bin";

    time_t start = time(NULL);
    
    ///////// VPS TEST //////////
    while(true)
    {
        
        // glClear(GL_COLOR_BUFFER_BIT);

        cv::Mat QueryImg;
        video >> QueryImg;
        if(QueryImg.empty()) {
            std::cout << " Finish Visual Localization " << std::endl; 
            break;
        }
        if (QueryImg.channels() > 1) cv::cvtColor(QueryImg, QueryImg, cv::COLOR_RGB2GRAY);
        std::cout << " Image Num is  :  " << image_num << "      !!!!!!!!!!!!!!!!!!!!" << std::endl;

            

        // Keyframe timestamp
        auto it = find(DB->timestamps.begin(),DB->timestamps.end(),timestamps[image_num]);
        if(it != DB->timestamps.end()){
            cv::Mat mask, QDescriptors;
            std::vector<cv::KeyPoint> QKeypoints;
            ORBfeatureAndDescriptorP3P(QueryImg, mask, QKeypoints, QDescriptors);

            // Place Recognition
            QueryResults ret;
            // ret.clear();

            std::vector<cv::Mat> VQDescriptors = MatToVectorMat(QDescriptors);
            db.query(VQDescriptors, ret, 20);
            std::cout << ret << std::endl;
            std::cout << "High score keyframe  num : "  << ret[0].Id << std::endl;
            // "       word num : " << ret[0].nWords << std::endl;
            // VPStest.SetCandidateKFid(ret);


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
            double PnPInlierRatio = VPStest.VPStestToReferenceKF(DB, QDescriptors, QKeypoints, ReferenceKFId, Pose, Inliers, Matches);
            std::cout << " PnPInlier Ratio of Selected Keyframe : " << PnPInlierRatio << std::endl;
            Eigen::Quaterniond q = ToQuaternion(Pose);
            
            // std::cout << "ReprojectioErr : ";
            // for(int i = 0; i < ReprojectionErr.size(); i++){
            //     std::cout << ReprojectionErr[i] << "  ";
            // }
            // std::cout << std::endl;
            
            std::cout << " Same As DataBase timestamp !!!  " << std::endl;            
                traj_file <<    timestamps[image_num] << " " << Pose(0, 3) << " " << Pose(1, 3) << " " << Pose(2, 3) << 
                " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
                             
                file << timestamps[image_num] <<  " " << PnPInlierRatio << " " << Inliers.rows << std::endl;



            test_file << timestamps[image_num] << " " << image_num << " " << Selected_KF_imageNum << " " << DBoW2Result_KF_imageNum << std::endl;
            
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

            cv::drawMatches(QueryImg, QKeypoints, DB->LeftKFimg[ReferenceKFId], DB2dMatchForDraw, Matches, MatchImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            // cv::Size a = MatchImg.size();
            cv::resize(MatchImg, MatchImg, cv::Size(1737, 376));
            cv::imshow("MatchImg", MatchImg);
                
            // VPS Result Pose
            show_trajectory_keyframe(Pose, 0.0, 0.0, 1.0, 0.5, 3.5);
            glFlush();

            // Save Result
            SaveVPStestResult->MatchingImg[KF_Img_Num] = MatchImg;
            // SaveVPStestResult->PnPpose[KF_Img_Num] = poseresult;
            SaveVPStestResult->PnPInlierRatio[KF_Img_Num] = PnPInlierRatio;
            SaveVPStestResult->PnPInliers[KF_Img_Num] = Inliers.rows;
            SaveVPStestResult->DBoW2ResultImgNum[KF_Img_Num] = Selected_KF_imageNum;
            // SaveVPStestResult->Inliers[KF_Img_Num] = Inliers;
            // SaveVPStestResult->ReprojectionErr.push_back(ReprojectionErr);
            // SaveVPStestResult->KFimageNum.push_back(KF_Img_Num);
            KF_Img_Num++;
            // cv::waitKey();
        }


        image_num++;
        std::cout << std::endl;

    }
    time_t finish = time(NULL);
    duration = (double)(finish - start);
    
    std::ofstream out(filepath, std::ios_base::binary);
    if (!out)
    {
        std::cout << "Cannot Write to Database File: " << std::endl;
        exit(-1);
    }
    boost::archive::binary_oarchive oa(out, boost::archive::no_header);
    oa << SaveVPStestResult;
    out.close();
    
    std::cout << "Total Landmark Num : " << DB->Landmarks.size() << std::endl;
    file << " Total VPS time is  : " << duration << " sec" << std::endl;
    std::cout << " Total VPS time is  : " << duration << " sec" << std::endl;
    traj_file.close();
    file.close();
    test_file.close();
    std::cout << " Finish VPS test " << std::endl;
    // cv::imshow("img", DB_images[0]);
    cv::waitKey();
    return 0;
}

     


