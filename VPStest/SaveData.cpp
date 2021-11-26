#include "BoostArchiver.h"
#include "DataBase.h"
#include "VPStest.h"
#include "ORBVocabulary.h"
#include "Parameter.h"
#include "Converter.h"

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
#include <sstream>
// #include <thread>
#include <pangolin/pangolin.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    std::vector<double> timestamps;
    std::vector<double> PnPInlierRatios;
    std::vector<Eigen::VectorXf> Pose;

    // Save PnPinlier result
    ofstream PnP_file;
    PnP_file.open(argv[1]);

    // Save trajectory result
    ofstream traj_file;
    traj_file.open(argv[2]);

    // Load timestamp and inlier ratio
    ifstream s;
    s.open("/home/donghoon/ORBSLAM2_MapCompression/VPStest/result/211124/plus_timestamp/Kitti00_original_pnp_inlier_ratio.txt");
    std::string line;

    while(std::getline(s, line)){

        
        std::string value;
        std::vector<std::string> values;
        std::stringstream ss(line);
        
        while(std::getline(ss, value, ' '))
            values.push_back(value);
    
        timestamps.push_back(std::stod(values[0]));
        PnPInlierRatios.push_back(std::stod(values[1]));
    }
    s.close();


    // Load pose
    ifstream s_;
    s_.open("/home/donghoon/ORBSLAM2_MapCompression/VPStest/result/211124/plus_timestamp/Kitti00_original_estimated_trajectory_result.txt");
    std::string line_;

    while(std::getline(s_, line_)){

        
        std::string value;
        std::vector<std::string> values;
        std::stringstream ss(line_);
        
        while(std::getline(ss, value, ' '))
            values.push_back(value);
        std::cout << values[7] << std::endl;
        Eigen::VectorXf p(7);
        p <<    std::stof(values[1]), std::stof(values[2]), std::stof(values[3]), 
                std::stof(values[4]), std::stof(values[5]), std::stof(values[6]), std::stof(values[7]);
        Pose.push_back(p);
    }
    s_.close();
std::cout << " a " << std::endl;
    // Write txt
    for(int i = 0; i < timestamps.size(); i++){

        if(PnPInlierRatios[i] > 0.6){

            PnP_file << timestamps[i] << " " << PnPInlierRatios[i] << std::endl;
            traj_file <<    timestamps[i] << " " << Pose[i](0) << " " << Pose[i](1) << " " << Pose[i](2) <<
                            " " << Pose[i](3) << " " << Pose[i](4) << " " << Pose[i](5) << " " << Pose[i](6) << std::endl;

        }

    }

    PnP_file.close();
    traj_file.close();
    
    return 0;
}