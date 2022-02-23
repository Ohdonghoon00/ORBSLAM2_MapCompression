#include "BoostArchiver.h"
#include "DataBase.h"
#include "VPStest.h"
#include "ORBVocabulary.h"
// #include "Parameter.h"
#include "Converter.h"
#include "Evaluation_euroc.h"
#include "utils.h"

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>


#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <cstdio>
#include <sstream>
// #include <thread>
// #include <pangolin/pangolin.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{



    // GT SLAM Pose txt, GT Query Pose txt
    std::string OriginalPosePath = argv[1];
    std::string CompressedPosePath = argv[2];
    
    std::ifstream OriginalPoseFile(OriginalPosePath, std::ifstream::in);
    std::ifstream CompressedPoseFile(CompressedPosePath, std::ifstream::in);
    
    if(!OriginalPoseFile.is_open()){
        std::cout << " GT SLAM Pose file failed to open " << std::endl;
        return EXIT_FAILURE;
    }    

    if(!CompressedPoseFile.is_open()){
        std::cout << " GT Query Pose file failed to open " << std::endl;
        return EXIT_FAILURE;
    }    

    int CompressedPoseLine_num(0), OriginalPoseLine_num;
    std::string CompressedPoseLine, OriginalPoseLine;
    
    std::vector<Vector6d> OriginalPoses, CompressedPoses;
    std::vector<double> IMUQuerytimestamps;
    
    // Original VPS Pose
    while(std::getline(OriginalPoseFile, OriginalPoseLine))
    {
        std::string OriginalPosevalue;
        std::vector<std::string> OriginalPosevalues;
        

        std::stringstream ss(OriginalPoseLine);
        while(std::getline(ss, OriginalPosevalue, ' '))
            OriginalPosevalues.push_back(OriginalPosevalue);        
        Vector6d Pose;
        Pose << std::stod(OriginalPosevalues[0]), std::stod(OriginalPosevalues[1]), std::stod(OriginalPosevalues[2]), 
                std::stod(OriginalPosevalues[3]), std::stod(OriginalPosevalues[4]), std::stod(OriginalPosevalues[5]);
        OriginalPoses.push_back(Pose);
        
        OriginalPoseLine_num++;
    }    
    
    // Compressed VPS Pose
    while(std::getline(CompressedPoseFile, CompressedPoseLine))
    {
        std::string CompressedPosevalue;
        std::vector<std::string> CompressedPosevalues;
        

        std::stringstream ss(CompressedPoseLine);
        while(std::getline(ss, CompressedPosevalue, ' '))
            CompressedPosevalues.push_back(CompressedPosevalue);
        Vector6d Pose;
        Pose << std::stod(CompressedPosevalues[0]), std::stod(CompressedPosevalues[1]), std::stod(CompressedPosevalues[2]), 
                std::stod(CompressedPosevalues[3]), std::stod(CompressedPosevalues[4]), std::stod(CompressedPosevalues[5]);
        CompressedPoses.push_back(Pose);
        
        CompressedPoseLine_num++;
    }

    std::string OriginalResultPath = argv[3];
    std::ifstream OriginalResultFile(OriginalResultPath, std::ifstream::in);
    std::vector<int> Inliers;

    std::string line1;
    while(std::getline(OriginalResultFile, line1)){
        std::string value;
        std::vector<std::string> values;

        std::stringstream ss(line1);
        while(std::getline(ss, value, ' '))
            values.push_back(value);
        Inliers.push_back(std::stoi(values[1]));
    }

    std::cout << "Data Load Finish" << std::endl;

    double RMS_Trans_Error(0.0), RMS_Rot_Error(0.0); 
    AbsoluteTrajectoryError(OriginalPoses, CompressedPoses, RMS_Trans_Error, RMS_Rot_Error);
    std::cout << " RMSE Trans Error : " << RMS_Trans_Error << std::endl;
    std::cout << " RMSE Rotation Error : " << Rad2Degree(RMS_Rot_Error) << std::endl;

    std::ofstream error_file;
    error_file.open("OriginalTo_70%_Error.txt");

    int TotalInliers = 0;
    int count = OriginalPoses.size();
    for(int i = 0; i < OriginalPoses.size(); i++){

    
        Eigen::Matrix4d GTMotion = To44RT(OriginalPoses[i]);
        Eigen::Matrix4d EsMotion = To44RT(CompressedPoses[i]);
        Eigen::Matrix4d RelativePose = GTMotion.inverse() * EsMotion;

        // trans
        Eigen::Vector3d RelativeTrans;
        RelativeTrans << RelativePose(0, 3), RelativePose(1, 3), RelativePose(2, 3);
        double RMS_Trans_Error_ = std::sqrt(RelativeTrans.dot(RelativeTrans));
        if(RMS_Trans_Error_ > 1.0){
            count--;
            TotalInliers += Inliers[i];
        }
        // rotation
        Eigen::Matrix3d RelativeRot_ = RelativePose.block<3, 3>(0, 0);
        Eigen::Vector3d RelativeRot = ToVec3(RelativeRot_);
        double RMS_Rot_Error_ = std::sqrt(RelativeRot.dot(RelativeRot));

        
        error_file << RMS_Trans_Error_ << " " << Rad2Degree(RMS_Rot_Error_) << std::endl;
    }

    std::cout << "Fail Inlier average : " << (double)TotalInliers / (double)(OriginalPoses.size() - count) << std::endl;
    std::cout << "Success Ratio : " << (double)count / (double)OriginalPoses.size() << std::endl;
    error_file.close();
    return 0;
}