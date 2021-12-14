#include "BoostArchiver.h"
#include "DataBase.h"
#include "VPStest.h"
#include "ORBVocabulary.h"
#include "Parameter.h"
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
    std::vector<double> timestamps;
    // std::vector<double> PnPInlierRatios;
    std::vector<Eigen::VectorXf> LcamGTPose, RcamGTPose;
    std::vector<Eigen::VectorXf> EstimatedPose;

    // Save PnPinlier result
    ofstream SelectedGTPose_file;
    SelectedGTPose_file.open("SelectedGTPose_original_result.txt");

    // Save trajectory result
    ofstream SelectedEsPose_file;
    SelectedEsPose_file.open("SelectedEsPose_original_result.txt");

    // Load GT Pose 
    ifstream s_;
    s_.open(argv[1]);
    std::string line_;

    while(std::getline(s_, line_)){

        
        std::string value;
        std::vector<std::string> values;
        std::stringstream ss(line_);
        
        while(std::getline(ss, value, ' '))
            values.push_back(value);
        Eigen::VectorXf Leftcam(7);
        Leftcam <<    std::stof(values[1]), std::stof(values[2]), std::stof(values[3]), 
                std::stof(values[4]), std::stof(values[5]), std::stof(values[6]), std::stof(values[7]);
        Eigen::VectorXf Rightcam(7);
        Rightcam = LeftCamToRightCam(Leftcam);
        RcamGTPose.push_back(Rightcam);
        LcamGTPose.push_back(Leftcam);
    }
    s_.close();
    
    // Load Estimated Pose
    ifstream s;
    s.open(argv[2]);
    std::string line;

    while(std::getline(s, line)){

        
        std::string value;
        std::vector<std::string> values;
        std::stringstream ss(line);
        
        while(std::getline(ss, value, ' '))
            values.push_back(value);
        
        Eigen::VectorXf p(7);
        p <<    std::stof(values[1]), std::stof(values[2]), std::stof(values[3]), 
                std::stof(values[4]), std::stof(values[5]), std::stof(values[6]), std::stof(values[7]);
        EstimatedPose.push_back(p);
        timestamps.push_back(std::stod(values[0]));    

    }
    s.close();
    
    // RMS ERROR
    std::cout << " ATE ... " << std::endl;
    float ATE_RMS_error = AbsoluteTrajectoryError(RcamGTPose, EstimatedPose);
    std::cout << " RPE ... " << std::endl;
    float RPE_RMS_error = RelativePoseError(RcamGTPose, EstimatedPose);

    // Write txt
    int count = 0;
    int Posethres = 10;
    for(int i = 0; i < timestamps.size(); i++){
        


        if(std::abs(RcamGTPose[i](2) - EstimatedPose[i](2)) < Posethres && std::abs(RcamGTPose[i](0) - EstimatedPose[i](0)) < Posethres && std::abs(RcamGTPose[i](1) - EstimatedPose[i](1)) < Posethres){
            
            
            SelectedGTPose_file << timestamps[i] << " " << RcamGTPose[i](0) << " " << RcamGTPose[i](1) << " " << RcamGTPose[i](2) << " "
            << RcamGTPose[i](3) << " " << RcamGTPose[i](4) << " " << RcamGTPose[i](5) << " " << RcamGTPose[i](6) << std::endl;
            
            SelectedEsPose_file << timestamps[i] << " " << EstimatedPose[i](0) << " " << EstimatedPose[i](1) << " " << EstimatedPose[i](2) << " "
            << EstimatedPose[i](3) << " " << EstimatedPose[i](4) << " " << EstimatedPose[i](5) << " " << EstimatedPose[i](6) << std::endl;            
            
            count++;

        }

    }
std::cout << count << std::endl;
std::cout << ATE_RMS_error << std::endl;
std::cout << RPE_RMS_error << std::endl;

    SelectedGTPose_file.close();
    SelectedEsPose_file.close();
    
    return 0;
}