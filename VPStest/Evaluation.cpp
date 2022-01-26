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
    std::vector<double> timestamps;
    // std::vector<double> PnPInlierRatios;
    std::vector<Eigen::VectorXf> LcamGTPose, RcamGTPose;
    std::vector<Eigen::VectorXf> EstimatedPose;

    // error
    ofstream TraslationError;
    TraslationError.open("Kitti00_TraslationError_20%_result.txt");

    // Save trajectory result
    // ofstream ;
    // SelectedGTPose_file.open("SelectedGTPose_original_result.txt");

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
        // timestamps.push_back(std::stod(values[0]));    

    }
    s.close();
    
    // RMS ERROR
    std::cout << " ATE ... " << std::endl;
    float ATE_RMS_error = AbsoluteTrajectoryError(LcamGTPose, EstimatedPose);
    std::cout << " RPE ... " << std::endl;
    float RPE_RMS_error = RelativePoseError(LcamGTPose, EstimatedPose);

    TranslationError(LcamGTPose, EstimatedPose, TraslationError);



std::cout << ATE_RMS_error << std::endl;
std::cout << RPE_RMS_error << std::endl;

    TraslationError.close();
    // SelectedEsPose_file.close();
    
    return 0;
}