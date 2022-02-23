#include "BoostArchiver.h"
#include "DataBase.h"
#include "VPStest.h"
#include "ORBVocabulary.h"
// #include "Parameter.h"
#include "Converter.h"
#include "Evaluation_euroc.h" 
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
    DataBase* DB;


    // Load DataBase for Visualize 2d point
    std::ifstream in(argv[1], std::ios_base::binary);
    if (!in){
        std::cout << "Cannot DataBase bin file is empty!" << std::endl;
        return false;
    }
    boost::archive::binary_iarchive ia(in, boost::archive::no_header);
    ia >> DB;
    in.close();
    
    // Load Map
    std::ifstream ins(argv[2], std::ios_base::binary);
    if (!ins){
        std::cout << "Cannot DataBase bin file is empty!" << std::endl;
        return false;
    }
    boost::archive::binary_iarchive ias(ins, boost::archive::no_header);
    ias >> VPStestResult;
    ins.close();

    // Original Map result TimeStamp and Selected KF
    std::string OriginalResultPath = argv[3];
    std::string OriginalImgNumPath = OriginalResultPath + "/MH01_MH02_VPStest_original_TimeStamp.txt";
    std::string OriginalResultSelPath = OriginalResultPath + "/MH01_MH02_original_Result.txt";
    
    ifstream OriginalImgNumFile(OriginalImgNumPath, std::ifstream::in);
    ifstream OriginalResultSelFile(OriginalResultSelPath, std::ifstream::in);

    std::vector<int> OriginalResultImgNum;
    std::vector<int> OriginalResultSelectedKF;

    std::string line1;
    while(std::getline(OriginalImgNumFile, line1)){
        std::string value;
        std::vector<std::string> values;

        std::stringstream ss(line1);
        while(std::getline(ss, value, ' '))
            values.push_back(value);
        OriginalResultImgNum.push_back(std::stoi(values[1]));
    }

    std::string line2;
    while(std::getline(OriginalResultSelFile, line2)){
        std::string value;
        std::vector<std::string> values;

        std::stringstream ss(line2);
        while(std::getline(ss, value, ' '))
            values.push_back(value);
        OriginalResultSelectedKF.push_back(std::stoi(values[3]));
    }
    
    int num = std::stoi(argv[4]);

    cv::Mat InlierMatchImg = VPStestResult->InlierMatchingImg[num];
    cv::imshow("InlierMatchImg", InlierMatchImg);

    cv::Mat TotalMatchImg = VPStestResult->TotalMatchingImg[num];
    cv::imshow("TotalMatchImg", TotalMatchImg); 

    auto it = std::find(OriginalResultImgNum.begin(), OriginalResultImgNum.end(), num);
    int index = it - OriginalResultImgNum.begin();
    int DBSelKFNum = OriginalResultSelectedKF[index];
    std::vector< cv::KeyPoint > KF2dPoints = DB->GetKF2dPoint(DBSelKFNum);
    cv::Mat img;
    cv::drawKeypoints(DB->LeftKFimg[DBSelKFNum], KF2dPoints, img);
    cv::imshow("DB_keypoints", img);
    // std::cout << "Inlier Ratio : " << VPStestResult->PnPInlierRatio[num] << std::endl;
    // std::cout << "Inliers : " << VPStestResult->PnPInliers[num] << std::endl;

    
    cv::waitKey();

    return 0;
}