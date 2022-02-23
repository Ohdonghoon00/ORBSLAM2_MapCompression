#include "BoostArchiver.h"
#include "DataBase.h"
#include "VPStest.h"
#include "ORBVocabulary.h"
// #include "Parameter.h"
#include "Converter.h"
#include "Evaluatation.h" 
#include "VPStestResult.h"

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

    VPStestResult* VPStestResult;
    
    // Load Map
    std::ifstream in(argv[1], std::ios_base::binary);
    if (!in){
        std::cout << "Cannot DataBase bin file is empty!" << std::endl;
        return false;
    }
    boost::archive::binary_iarchive ia(in, boost::archive::no_header);
    ia >> VPStestResult;
    in.close();

    int num = std::stoi(argv[2]);

    cv::Mat MatchImg = VPStestResult->MatchingImg[num];
    cv::imshow("MatchImg", MatchImg); 

    std::cout << "Inlier Ratio : " << VPStestResult->PnPInlierRatio[num] << std::endl;
    std::cout << "Inliers : " << VPStestResult->PnPInliers[num] << std::endl;

    
    cv::waitKey();

    return 0;
}