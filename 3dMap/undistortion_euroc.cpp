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


using namespace std;
using namespace cv;

void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps);

int main(int argc, char **argv)
{

    // Read rectification parameters
    cv::FileStorage fsSettings(argv[3], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
            rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return -1;
    }

    cv::Mat M1l,M2l;
    cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);

    // Load timestamp
    std::string timestampPath = argv[1];
    ifstream s;
    s.open(timestampPath);
    std::string line;
    std::vector<cv::Mat> Undistortimgs;

    // Load Query Img
    std::cout << " Input Query Img " << std::endl;
    std::string QueryPath = argv[2];
    
    while(std::getline(s, line)){
        
        std::stringstream ss;
        ss << line;
        std::string QueryImgsPath = QueryPath + "/" + ss.str() +".png";
        cv::Mat img = cv::imread(QueryImgsPath);
        std::string RectimgName_ = "/home/donghoon/ORBSLAM2_MapCompression/3dMap/DataBase/Euroc/MH03/RectCam1";
        std::string RectimgName = RectimgName_ + "/" + ss.str() +".png";
        cv::Mat imLeftRect;
        cv::remap(img,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
        cv::imwrite(RectimgName, imLeftRect );
        cv::imshow("originalImg", img);
        cv::imshow("RectImg", imLeftRect);
        // cv::waitKey();
        std::cout << line << std::endl;

    }
    
    s.close();


    std::cout << Undistortimgs.size() << std::endl;


    return 0;
}