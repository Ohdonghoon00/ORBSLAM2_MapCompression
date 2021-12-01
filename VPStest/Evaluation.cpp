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
#include <pangolin/pangolin.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    std::vector<double> timestamps;
    // std::vector<double> PnPInlierRatios;
    std::vector<Eigen::VectorXf> GTPose;
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
        Eigen::VectorXf p(7);
        p <<    std::stof(values[1]), std::stof(values[2]), std::stof(values[3]), 
                std::stof(values[4]), std::stof(values[5]), std::stof(values[6]), std::stof(values[7]);
        GTPose.push_back(p);
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
    float ATE_RMS_error = AbsoluteTrajectoryError(GTPose, EstimatedPose);
    std::cout << " RPE ... " << std::endl;
    float RPE_RMS_error = RelativePoseError(GTPose, EstimatedPose);

    // Write txt
    int count = 0;
    for(int i = 0; i < timestamps.size(); i++){
        


        if(std::abs(GTPose[i](2) - EstimatedPose[i](2)) < 5 && std::abs(GTPose[i](0) - EstimatedPose[i](0)) < 5 && std::abs(GTPose[i](1) - EstimatedPose[i](1)) < 5){
            
            
            SelectedGTPose_file << timestamps[i] << " " << GTPose[i](0) << " " << GTPose[i](1) << " " << GTPose[i](2) << " "
            << GTPose[i](3) << " " << GTPose[i](4) << " " << GTPose[i](5) << " " << GTPose[i](6) << std::endl;
            
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