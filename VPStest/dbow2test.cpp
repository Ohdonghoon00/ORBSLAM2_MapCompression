#include "BoostArchiver.h"
#include "DataBase.h"
#include "VPStest.h"
#include "ORBVocabulary.h"
#include "Parameter.h"
#include "Converter.h"
#include "ORBextractor.h"
#include "map_viewer.h"

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
// #include <pangolin/pangolin.h>

using namespace std;
using namespace cv;
using namespace DBoW2;

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

int main(int argc, char **argv)
{
    VPStest DBVPStest(500), QVPStest(500);
    std::vector< std::vector<cv::Mat > > features;
    
    int nFeatures = 4000;
    float scaleFactor = 1.2;
    int nlevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;
    
    int DB_image = 0;
    int Q_image = 0;

    std::vector<std::vector<cv::KeyPoint>> TotalDBKeypoints;
    std::vector<cv::Mat> TotalDBDescriptors;

    std::vector<DBoW2::BowVector> DBBoWVectors;
    // Load voc
    std::cout << "load voc" << std::endl;
    ORBVocabulary voc(argv[1]);
    // ORBVocabulary voc;
    // voc.loadFromTextFile(argv[1]);
    std::cout << "copy voc to db" << std::endl;
    OrbDatabase db(voc, false, 0); // false = do not use direct index



    std::string DataPath = argv[2];

    // Save DB
    std::cout << " save db Img " << std::endl;
    std::string DBPath = DataPath + "/image_0/%06d.png";
    cv::VideoCapture video;
    if(!video.open(DBPath)){
        std::cout << " No db image " << std::endl;
        return -1;
    }    

    while(true){
        cv::Mat dbImg;
        video >> dbImg;
        if(dbImg.empty()) {
            std::cout << " DB " << std::endl; 
            break;
        }
        if (dbImg.channels() > 1) cv::cvtColor(dbImg, dbImg, cv::COLOR_RGB2GRAY);
        if(DB_image % 5 == 0){
        std::cout << " db Image Num is  :  " << DB_image << "      !!!!!!!!!!!!!!!!!!!!" << std::endl;
        
    
        cv::Mat mask, DBDescriptors;
        std::vector<cv::KeyPoint> DBKeypoints;
        
        ORBextractor ORBfeatureAndDescriptor(500, scaleFactor, nlevels, iniThFAST, minThFAST);
        ORBfeatureAndDescriptor(dbImg, mask, DBKeypoints, DBDescriptors);
        
        // DBKeypoints = DBVPStest.ORBFeatureExtract(dbImg);
        // DBDescriptors = DBVPStest.ORBDescriptor(dbImg, DBKeypoints);        
        
        TotalDBKeypoints.push_back(DBKeypoints);
        TotalDBDescriptors.push_back(DBDescriptors);
        std::vector<cv::Mat> VDBDescriptors = MatToVectorMat(DBDescriptors);
        DBoW2::BowVector bowVector;
        DBoW2::FeatureVector bowFeaturevector;
        voc.transform(VDBDescriptors, bowVector, bowFeaturevector, 4);
        DBBoWVectors.push_back(bowVector);

        // features.clear();
        // features.push_back(std::vector<cv::Mat >());
        // changeStructure(DBDescriptors, features.back());
        
        db.add(VDBDescriptors);
    
        }
        DB_image++;
        if(DB_image > 500) break;
    }

    // Load Query Img
    std::cout << " Input Query Img " << std::endl;
    std::string QueryPath = DataPath + "/image_1/%06d.png";
    cv::VideoCapture video_;
    if(!video_.open(QueryPath)){
        std::cout << " No query image " << std::endl;
        return -1;
    }

    while(true){
        
        cv::Mat QueryImg;
        video_ >> QueryImg;
        if(QueryImg.empty()) {
            std::cout << " Finish Visual Localization " << std::endl; 
            break;
        }
        if (QueryImg.channels() > 1) cv::cvtColor(QueryImg, QueryImg, cv::COLOR_RGB2GRAY);
        std::cout << "query image num : " << Q_image << std::endl;
        QueryResults ret;
        ret.clear();
        
        cv::Mat mask, QDescriptors;
        std::vector<cv::KeyPoint> QKeypoints;
        
        ORBextractor ORBfeatureAndDescriptor_(500, scaleFactor, nlevels, iniThFAST, minThFAST);
        ORBfeatureAndDescriptor_(QueryImg, mask, QKeypoints, QDescriptors);
        
        // QKeypoints = QVPStest.ORBFeatureExtract(QueryImg);
        // QDescriptors = QVPStest.ORBDescriptor(QueryImg, QKeypoints);
        
        std::vector<cv::Mat> VQDescriptors = MatToVectorMat(QDescriptors);
        
        
        
        DBoW2::BowVector Qbowvector;
        DBoW2::FeatureVector QbowFeaturevector;

        voc.transform(VQDescriptors, Qbowvector, QbowFeaturevector, 4);

        // features.clear();
        // features.push_back(std::vector<cv::Mat >());
        // changeStructure(QDescriptors, features.back());

        db.query(VQDescriptors, ret, 20);
        std::cout << ret << std::endl;
        int BestScoreNum = ret[0].Id * 5;
        float score = voc.score(DBBoWVectors[ret[0].Id], Qbowvector);
        std::cout << "score : " << score << std::endl; 
        std::stringstream ShowDBimagePath;  
        ShowDBimagePath << DataPath + "/image_0/" << std::setfill('0') << std::setw(6) << BestScoreNum << ".png";
        cv::Mat DB_image = cv::imread(ShowDBimagePath.str(), cv::ImreadModes::IMREAD_GRAYSCALE);
        std::vector<cv::DMatch> matches = QVPStest.ORBDescriptorMatch(QDescriptors, TotalDBDescriptors[BestScoreNum]);
        std::sort(matches.begin(), matches.end());
        std::vector<cv::DMatch> goodmatch(matches.begin(), matches.begin() + 100);
        cv::Mat showimage1, showimage2, showimage;
        cv::drawKeypoints(QueryImg, QKeypoints, showimage1);
        cv::drawKeypoints(DB_image, TotalDBKeypoints[BestScoreNum], showimage2);
        cv::drawMatches(QueryImg, QKeypoints, DB_image, TotalDBKeypoints[BestScoreNum], goodmatch, showimage);
        cv::imshow("matchimage", showimage);
        cv::imshow("image1", showimage1);
        cv::imshow("image2", showimage2);
        
        Q_image++;
        cv::waitKey();
    }

    return 0;
}