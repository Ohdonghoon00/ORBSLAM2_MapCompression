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
// #include <thread>
#include <pangolin/pangolin.h>

using namespace std;
using namespace cv;
using namespace DBoW2;

int main(int argc, char **argv)
{

    DataBase DB;
    int image_num = 0;


    // Load Map
    std::ifstream in(argv[1], std::ios_base::binary);
    if (!in){
        std::cout << "Cannot DataBase bin file is empty!" << std::endl;
        return false;
    }
    boost::archive::binary_iarchive ia(in, boost::archive::no_header);
    ia >> DB;
    in.close();

    // Load voc
    std::cout << "load voc" << std::endl;
    ORBVocabulary voc;
    voc.loadFromTextFile("../Vocabulary/ORBvoc.txt");
    std::cout << "copy voc to db" << std::endl;
    OrbDatabase db(voc, false, 0); // false = do not use direct index

    // Input DB descriptor to voc
    for(size_t i = 0; i < DB.KFtoMPIdx.size(); i++){
        std::vector<cv::Mat> KFDescriptor;
        KFDescriptor.clear();
        KFDescriptor = MatToVectorMat(DB.GetKFMatDescriptor(i));
        std::cout << " KF num : " << i << "     Keypoint num : " << (DB.GetKFMatDescriptor(i)).size() << std::endl;  
        db.add(KFDescriptor);
    }

    // Load timestamp
    ifstream s;
    s.open("../times.txt");
    std::string line;
    std::vector<double> timestamps;

    while(std::getline(s, line))
        timestamps.push_back(std::stod(line));
    
    s.close();
    
    // Load Query Img
    std::cout << " Input Query Img " << std::endl;
    std::string QueryPath = "/media/donghoon/1462-5978/kitti/sequences/00/image_1/%06d.png";
    // std::string QueryPath = "/media/donghoon/1462-5978/kitti/sequences/00/image_0/000095.png";
    cv::VideoCapture video;
    if(!video.open(QueryPath)){
        std::cout << " No query image " << std::endl;
        return -1;
    }

    // Save PnPinlier result
    ofstream file;
    file.open(argv[2]);

    // Save trajectory result
    ofstream traj_file;
    traj_file.open(argv[3]);

    ///////// VPS TEST //////////
    while(true)
    {
        cv::Mat QueryImg;
        video >> QueryImg;
        if(QueryImg.empty()) {
            std::cout << " Finish Visual Localization " << std::endl; 
            break;
        }
        if (QueryImg.channels() > 1) cv::cvtColor(QueryImg, QueryImg, cv::COLOR_RGB2GRAY);
        std::cout << " Image Num is  :  " << image_num << "      !!!!!!!!!!!!!!!!!!!!" << std::endl;
        VPStest VPStest;

        // Extract ORB Feature and Destriptor
        std::cout << " Extract ORB Feature and Descriptor " << std::endl;
        std::vector<cv::KeyPoint> QKeypoints;
        cv::Mat QDescriptors;
        QKeypoints = VPStest.ORBFeatureExtract(QueryImg);
        QDescriptors = VPStest.ORBDescriptor(QueryImg, QKeypoints);

        // Place Recognition
        QueryResults ret;
        ret.clear();

        std::vector<cv::Mat> VQDescriptors = MatToVectorMat(QDescriptors);
        db.query(VQDescriptors, ret, 20);
        // std::cout << ret << std::endl;
        std::cout << "High score keyframe  num : "  << ret[0].Id << std::endl;
        // "       Score : " << ret[0].Score << std::endl;
        VPStest.SetCandidateKFid(ret);

        std::cout << std::endl;
        
        // FindReferenceKF
        std::cout << "Find Reference Keyframe !! " << std::endl;
        int ReferenceKFId = VPStest.FindReferenceKF(DB, QDescriptors, QKeypoints);
        std::cout << " Selected Keyframe num : " << ReferenceKFId << std::endl;

        // VPS test to ReferenceKF
        Eigen::Matrix4f Pose;
        double PnPInlierRatio = VPStest.VPStestToReferenceKF(DB, QDescriptors, QKeypoints, ReferenceKFId, Pose);
        std::cout << " PnPInlier Ratio of Selected Keyframe : " << PnPInlierRatio << std::endl;
        Eigen::Quaternionf q = ToQuaternion(Pose);

        std::cout << Pose << std::endl;
        // Save timestamp + trajectory
        auto it = find(DB.timestamps.begin(),DB.timestamps.end(),timestamps[image_num]);
        if(it != DB.timestamps.end()){
            traj_file <<    timestamps[image_num] << " " << Pose(0, 3) << " " << Pose(1, 3) << " " << Pose(2, 3) << " " <<
                        q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            file << timestamps[image_num] <<  " " << PnPInlierRatio << std::endl;
        }


     

        // Storage pose and inlier ratio


        image_num++;
    
    }
    traj_file.close();
    file.close();
    std::cout << " Finish VPS test " << std::endl;
    return 0;
}