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
    std::string GTSLAMPosePath = argv[1];
    std::string GTQueryPosePath = argv[2];
    
    std::ifstream GTSLAMPoseFile(GTSLAMPosePath, std::ifstream::in);
    std::ifstream GTQueryPoseFile(GTQueryPosePath, std::ifstream::in);
    
    if(!GTSLAMPoseFile.is_open()){
        std::cout << " GT SLAM Pose file failed to open " << std::endl;
        return EXIT_FAILURE;
    }    

    if(!GTQueryPoseFile.is_open()){
        std::cout << " GT Query Pose file failed to open " << std::endl;
        return EXIT_FAILURE;
    }    

    int GTQueryPoseLine_num(0), GTSLAMPoseLine_num;
    std::string GTQueryPoseLine, GTSLAMPoseLine;
    
    std::vector<Vector6d> GTSLAMPoses, GTQueryPoses;
    std::vector<double> IMUQuerytimestamps;
    
    // GT SLAM Pose
    while(std::getline(GTSLAMPoseFile, GTSLAMPoseLine))
    {
        if(GTSLAMPoseLine_num == 0){
            GTSLAMPoseLine_num++;
            continue;
        }
        std::string GTSLAMPosevalue;
        std::vector<std::string> GTSLAMPosevalues;
        

        std::stringstream ss(GTSLAMPoseLine);
        while(std::getline(ss, GTSLAMPosevalue, ' '))
            GTSLAMPosevalues.push_back(GTSLAMPosevalue);        
        
        Eigen::Quaterniond q;
        q.x() = std::stod(GTSLAMPosevalues[4]);
        q.y() = std::stod(GTSLAMPosevalues[5]);
        q.z() = std::stod(GTSLAMPosevalues[6]);
        q.w() = std::stod(GTSLAMPosevalues[7]);
        Eigen::Vector3d t;
        t << std::stod(GTSLAMPosevalues[1]), std::stod(GTSLAMPosevalues[2]),std::stod( GTSLAMPosevalues[3]);
        Vector6d Pose = To6DOF(q, t);
        GTSLAMPoses.push_back(Pose);
        
        GTSLAMPoseLine_num++;
    }    
    
    // GT Query Pose
    while(std::getline(GTQueryPoseFile, GTQueryPoseLine))
    {
        if(GTQueryPoseLine_num == 0){
            GTQueryPoseLine_num++;
            continue;
        }
        std::string GTQueryPosevalue;
        std::vector<std::string> GTQueryPosevalues;
        

        std::stringstream ss(GTQueryPoseLine);
        while(std::getline(ss, GTQueryPosevalue, ' '))
            GTQueryPosevalues.push_back(GTQueryPosevalue);
        
        
        IMUQuerytimestamps.push_back(std::stod(GTQueryPosevalues[0]));
        Eigen::Quaterniond q;
        q.x() = std::stod(GTQueryPosevalues[4]);
        q.y() = std::stod(GTQueryPosevalues[5]);
        q.z() = std::stod(GTQueryPosevalues[6]);
        q.w() = std::stod(GTQueryPosevalues[7]);
        Eigen::Vector3d t;
        t << std::stod(GTQueryPosevalues[1]), std::stod(GTQueryPosevalues[2]), std::stod(GTQueryPosevalues[3]);
        Vector6d Pose = To6DOF(q, t);
        GTQueryPoses.push_back(Pose);
        
        GTQueryPoseLine_num++;
    }

    // camera timestamp
    ifstream s;
    std::vector<double> Camtimestamps;
    s.open(argv[3]);
    std::string line;

    while(std::getline(s, line)){
        Camtimestamps.push_back(std::stod(line) * 10e-10);
    }
    s.close();

    // Transformation    
    Eigen::Matrix4d GTSLAM44Poses = To44RT(GTSLAMPoses[0]);
    Eigen::Matrix4d Cam2Body = GetCam2Body(Cam2BodyData);
    std::cout << Cam2Body << std::endl;
    Eigen::Matrix4d camSLAMPoses = GTSLAM44Poses * Cam2Body;

    std::vector<Vector6d> QueryEvposes;
    for(int i = 0; i < GTQueryPoses.size(); i++){
      Eigen::Matrix4d pose_ = camSLAMPoses.inverse() * To44RT(GTQueryPoses[i]) * Cam2Body;
      Vector6d pose = To6DOF(pose_);
      QueryEvposes.push_back(pose);
    }



    // Save file
    ofstream file;
    file.open("MH01_MH02_GTcam_Trajectory.txt");
    
    for(int i = 0; i < Camtimestamps.size(); i++){
        double Min = std::numeric_limits<double>::max();
        int idx = -1;
        
        for(int j = 0; j < IMUQuerytimestamps.size(); j++){
            double diff = std::fabs(Camtimestamps[i] - IMUQuerytimestamps[j]);
            if(diff < Min){
                Min = diff;
                idx = j;
            }
        }
        // std::cout << setprecision(20) << Camtimestamps[i] << "   " << IMUQuerytimestamps[idx] << std::endl;

        file << setprecision(20) << Camtimestamps[i] << " " << setprecision(9) << QueryEvposes[idx][0] << " " << QueryEvposes[idx][1] << " " << QueryEvposes[idx][2] << 
                " " << QueryEvposes[idx][3] << " " << QueryEvposes[idx][4] << " " << QueryEvposes[idx][5] << std::endl;
    }

    file.close();


    return 0;
}