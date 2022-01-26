#include "BoostArchiver.h"
#include "DataBase.h"
#include "VPStest.h"
#include "ORBVocabulary.h"
// #include "Parameter.h"
#include "Converter.h"
#include "Evaluatation.h" 

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

    // // GT DB Pose csv
    // std::string GTDBPosePath = argv[1];
    // std::ifstream GTDBPoseFile(GTDBPosePath, std::ifstream::in);

    // if(!GTDBPoseFile.is_open()){
    //     std::cout << " GT DB Pose file failed to open " << std::endl;
    //     return EXIT_FAILURE;
    // }    

    // int GTDBPoseLine_num = 0;
    // std::string GTDBPoseLine;
    
    // while(std::getline(GTDBPoseFile, GTDBPoseLine) && ros::ok())
    // {
    //     if(GTDBPoseLine_num == 0){
    //         GTDBPoseLine_num++;
    //         continue;
    //     }

    // GT Query Pose csv
    std::string GTQueryPosePath = argv[1];
    std::ifstream GTQueryPoseFile(GTQueryPosePath, std::ifstream::in);

    if(!GTQueryPoseFile.is_open()){
        std::cout << " GT Query Pose file failed to open " << std::endl;
        return EXIT_FAILURE;
    }    

    int GTQueryPoseLine_num = 0;
    std::string GTQueryPoseLine;
    std::vector<Vector6d> Poses;
    std::vector<double> IMUtimestamps;
    std::vector<double> Camtimestamps;

    while(std::getline(GTQueryPoseFile, GTQueryPoseLine))
    {
        if(GTQueryPoseLine_num == 0){
            GTQueryPoseLine_num++;
            continue;
        }
        
        std::string GTQueryPosevalue;
        std::vector<std::string> GTQueryPosevalues;

        std::stringstream ss(GTQueryPoseLine);
        while(std::getline(ss, GTQueryPosevalue, ','))
            GTQueryPosevalues.push_back(GTQueryPosevalue);
        IMUtimestamps.push_back(std::stod(GTQueryPosevalues[0]));
        Eigen::Quaterniond q;
        q.x() = std::stod(GTQueryPosevalues[5]);
        q.y() = std::stod(GTQueryPosevalues[6]);
        q.z() = std::stod(GTQueryPosevalues[7]);
        q.w() = std::stod(GTQueryPosevalues[4]);

        Eigen::Vector3d r = To3DOF(q);
        Vector6d Pose;
        Pose << r.x(), r.y(), r.z(), std::stod(GTQueryPosevalues[1]), std::stod(GTQueryPosevalues[2]), std::stod(GTQueryPosevalues[3]);
        Poses.push_back(Pose);
    }

    // camera timestamp
    ifstream s;
    s.open(argv[2]);
    std::string line;

    while(std::getline(s, line)){
        Camtimestamps.push_back(std::stod(line));
    }
    s.close();
        



    // Save file
    ofstream file;
    file.open("MH02_GT_Trajectory.txt");
    Eigen::Matrix4d RT = To44RT(Poses[0]);
    
    for(int i = 0; i < Camtimestamps.size(); i++){
        double Min = std::numeric_limits<double>::max();
        int idx = -1;
        
        for(int j = 0; j < IMUtimestamps.size(); j++){
            double diff = std::fabs(Camtimestamps[i] - IMUtimestamps[j]);
            if(diff < Min){
                Min = diff;
                idx = j;
            }
        }
        std::cout << setprecision(20) << Camtimestamps[i] << "   " << IMUtimestamps[idx] << std::endl;
        Eigen::Matrix4d popo = RT.inverse() * To44RT(Poses[idx]);
        Eigen::Quaterniond qqq = ToQuaternion(popo);
        file << i << " " << popo(0, 3) << " " << popo(1, 3) << " " << popo(2, 3) << 
                " " << qqq.x() << " " << qqq.y() << " " << qqq.z() << " " << qqq.w() << std::endl;
    }

    file.close();


    return 0;
}