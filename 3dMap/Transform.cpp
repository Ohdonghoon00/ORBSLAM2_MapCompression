#include <iostream>
#include<algorithm>
#include<fstream>
#include <sstream>
#include <cmath>

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

#include "BoostArchiver.h"
#include "DataBase.h"
#include "Parameter.h"
#include "Converter.h"
#include "ORBextractor.h"
#include "map_viewer.h"
#include "utils.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    std::vector<double> CamTimestamps;
    // Load Cam Timestamps
    std::string CamTimeStampPath = argv[1];
    std::ifstream CamTimeStampFile(CamTimeStampPath, std::ifstream::in);
    if(!CamTimeStampFile.is_open()){
        std::cout << " Cam timestamp file failed to open " << std::endl;
        return EXIT_FAILURE;
    }

    std::string CamTimestampLine;
    while(std::getline(CamTimeStampFile, CamTimestampLine))
    {
        CamTimestamps.push_back(std::stod(CamTimestampLine));
    }
    
    // Load GtIMUPose
    std::string GtImuPosePath = argv[2];
    std::ifstream GtImuPoseFile(GtImuPosePath, std::ifstream::in);
    if(!GtImuPoseFile.is_open()){
        std::cout << " GT IMU Pose file failed to open " << std::endl;
        return EXIT_FAILURE;
    }
    
    int GtImuPoseLine_num = 0;
    std::string GtImuPoseLine;
    std::vector<Vector6d> GTImuPoses;
    std::vector<double> ImuTimestamps;
    
    while(std::getline(GtImuPoseFile, GtImuPoseLine))
    {
        if(GtImuPoseLine_num == 0){
            GtImuPoseLine_num++;
            continue;
        }
        std::string value;
        std::vector<std::string> values;
        

        std::stringstream ss(GtImuPoseLine);
        while(std::getline(ss, value, ','))
            values.push_back(value);
       
        

        Eigen::Quaterniond q;
        q.x() = std::stod(values[5]);
        q.y() = std::stod(values[6]);
        q.z() = std::stod(values[7]);
        q.w() = std::stod(values[4]);

        Eigen::Vector3d t;
        t << std::stod(values[1]), std::stod(values[2]),std::stod( values[3]);
        Vector6d Pose = To6DOF(q, t);
        GTImuPoses.push_back(Pose);
        // double timestamp = std::floor(std::stod(values[0]) * 1e5) * 1e-5;
        ImuTimestamps.push_back(std::stod(values[0]));

        GtImuPoseLine_num++;
    }  
    
    

    // Save Pose
    std::ofstream gtCamPose;
    gtCamPose.open("MH03_Cam1Pose.txt");
    
    for(int i = 0; i < CamTimestamps.size(); i++){

        int Minidx = -1;
        double Minvalue = DBL_MAX;
        for(int j = 0; j < ImuTimestamps.size(); j++){
        
            double diff = std::fabs(CamTimestamps[i] - ImuTimestamps[j]);
            if(diff < Minvalue){
                Minvalue = diff;
                Minidx = j;
            }
        }

        if(Minvalue > 1000) continue;

        std::string CamTimestampsImgPath = std::to_string(int64_t(CamTimestamps[i]));
        std::string imgPath = "/home/donghoon/ORBSLAM2_MapCompression/3dMap/DataBase/Euroc/MH03/RectCam1/" + CamTimestampsImgPath + ".png";
        cv::Mat image = cv::imread(imgPath);
        
        std::string SortImgPath = "/home/donghoon/ORBSLAM2_MapCompression/3dMap/DataBase/Euroc/MH03/RectCam1Sort/" + CamTimestampsImgPath + ".png";
        cv::imwrite(SortImgPath, image );
        
        std::cout << i << " " << Minidx << " " << Minvalue << std::endl;

        Eigen::Matrix4d GtImu44poses = To44RT(GTImuPoses[Minidx]);
        Eigen::Matrix4d Cam2Body = GetCam2Body(Cam1ToBodyData);
        Eigen::Matrix4d camSLAMPoses = GtImu44poses * Cam2Body;        
        Vector6d pose = To6DOF(camSLAMPoses);

        gtCamPose << CamTimestampsImgPath << " " << pose[0] << " " << pose[1] << " " << pose[2] << " " << pose[3] << " " << pose[4] << " " << pose[5] << std::endl;
    }
        // for(int j = 0; j < ImuTimestamps.size(); j++){
        

        //     // if(j == 36305)
        //         std::cout << j << " " << setprecision(20) << ImuTimestamps[j] << std::endl;
        //     // std::cout << setprecision(20) << CamTimestamps[i] << std::endl;
        // }
    return 0;
}


