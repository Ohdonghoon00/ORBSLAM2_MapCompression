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
    std::vector<Vector6d> GTImuPoses;
    std::vector<double> ImuTimestamps;
    readCsvGtPose(GtImuPosePath, &GTImuPoses, &ImuTimestamps);

    
    // db(MH01) initial pose
    std::string dbInitialPath = argv[3];
    std::vector<Vector6d> dbInitialPoses;
    std::vector<double> dbInitialTimes;
    readCsvGtPose(dbInitialPath, &dbInitialPoses, &dbInitialTimes);
    
    Eigen::Matrix4d Cam2Body = GetCam2Body(Cam0ToBodyData);
    Eigen::Matrix4d initialPose = To44RT(dbInitialPoses[5]) * Cam2Body;
    
    
    // Save Pose
    std::ofstream gtCamPose;
    gtCamPose.open("MH01db_MH01cam0Pose.txt");
    std::ofstream gtTime;
    gtTime.open("MH01_timeStamp.txt");
    int cnt = 0;
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
        std::string imgPath = "/home/ohdonghoon/EuroC/MH01/RectCam0/" + CamTimestampsImgPath + ".png";
        cv::Mat image = cv::imread(imgPath);
        
        std::string SortImgPath = "/home/ohdonghoon/EuroC/MH01/RectCam0_for_EsPose/" + CamTimestampsImgPath + ".png";
        cv::imwrite(SortImgPath, image );
        
        std::cout << i << " " << Minidx << " " << Minvalue << std::endl;
        Eigen::Matrix4d GtImu44poses = To44RT(GTImuPoses[Minidx]);
        Eigen::Matrix4d camSLAMPoses = initialPose.inverse() * GtImu44poses * Cam2Body;        
        Vector6d pose = To6DOF(camSLAMPoses);

        gtCamPose << CamTimestampsImgPath << " " << pose[0] << " " << pose[1] << " " << pose[2] << " " << pose[3] << " " << pose[4] << " " << pose[5] << std::endl;
        gtTime << CamTimestampsImgPath << std::endl;
        cnt++;
    }
        // for(int j = 0; j < ImuTimestamps.size(); j++){
        

        //     // if(j == 36305)
        //         std::cout << j << " " << setprecision(20) << ImuTimestamps[j] << std::endl;
        //     // std::cout << setprecision(20) << CamTimestamps[i] << std::endl;
        // }
    return 0;
}


