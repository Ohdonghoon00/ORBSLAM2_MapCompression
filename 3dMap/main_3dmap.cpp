#include <iostream>
#include<algorithm>
#include<fstream>
#include <sstream>
#include <cmath>

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/xfeatures2d.hpp"
// #include <opencv2/nonfree/nonfree.hpp>
// #include <opencv2/nonfree/features2d.hpp>

#include <Eigen/Dense>

#include "BoostArchiver.h"
#include "DataBase.h"
#include "Parameter.h"
#include "Converter.h"
#include "ORBextractor.h"
#include "map_viewer.h"
#include "utils.h"
#include "Map.h"
#include "Keyframe.h"
#include "BA.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    //
    cv::VideoCapture video;
    std::vector<Vector6d> KFgtPoses;
    int KFcnt = 0;
    cv::Mat K = GetK(IntrinsicData);
    std::cout << K << std::endl;
    
    // Extract feature
    int dbow2Features = 6000;
    int nFeatures = 5000;
    float scaleFactor = 1.2;
    int nlevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;
    
    int max_keypoint = 5000;
    int minDis = 4;
    int maxIds = 0;
    double KeyframeTrackRatio = 0.60;
    int trackPtsNum = 50;
    int trackFrameNum = 20;
    int currTrackFrameNum = 0;
    double lossfuncParameter = 1 / ((fx + fy) / 2);
    // Viewer
    glutInit(&argc, argv);
    initialize_window();

    ORBextractor ORBfeatureAndDescriptor(dbow2Features, scaleFactor, nlevels, iniThFAST, minThFAST);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(dbow2Features);
    cv::Ptr<Feature2D> sift = cv::xfeatures2d::SIFT::create(nFeatures, 5, 0.025, 1000.0);

    //
    Keyframe Curr_KF;
    Map MapDB;
    Track Track;
    std::vector<Keyframe> KFdb;
    std::map< int, std::vector<cv::DMatch>> matches;
    int GoodMatchNum = 1000;

    std::string KFgtPosePath = argv[3];
    ReadgtPose(KFgtPosePath, &KFgtPoses);
    
    // Load timestamp
    std::string timestampPath = argv[1];
    ifstream s;
    s.open(timestampPath);
    std::string line;
    std::vector<double> timestamps;
    std::vector<cv::Mat> Queryimgs1;
    std::vector<cv::Mat> Queryimgs2;
    // Load Query Img
    std::cout << " Input Query Img " << std::endl;
    std::string QueryPath = argv[2];

    while(std::getline(s, line)){
        
        std::stringstream ss;
        ss << line;
        std::string QueryImgsPath1 = QueryPath + "/RectCam0_for_EsPose/" + ss.str() +".png";
        cv::Mat image1 = cv::imread(QueryImgsPath1);
            std::string QueryImgsPath2 = QueryPath + "/RectCam1_for_EsPose/" + ss.str() +".png";
        cv::Mat image2 = cv::imread(QueryImgsPath2);

        timestamps.push_back(std::stod(line) * 10e-10);
    
        if (image1.channels() > 1) cv::cvtColor(image1, image1, cv::COLOR_RGB2GRAY);
        if (image2.channels() > 1) cv::cvtColor(image2, image2, cv::COLOR_RGB2GRAY);
    
     
        std::cout << " KF img Num : " << KFcnt << "  @@@@@@@@@@@@@@@@@ " << std::endl;


/////////// stereo KLT initial /////////////
        if(KFcnt == 0){
            
            Curr_KF.EraseClass();
            Curr_KF.limage = image1.clone();
            Curr_KF.rimage = image2.clone();
            Curr_KF.timeStamp = timestamps[KFcnt];
            
            // test sift
            sift->detectAndCompute(Curr_KF.limage, Curr_KF.lmask, Curr_KF.lKeyPoints, Curr_KF.lDescriptors);
            sift->detectAndCompute(Curr_KF.rimage, Curr_KF.rmask, Curr_KF.rKeyPoints, Curr_KF.rDescriptors);
            // ORBfeatureAndDescriptor(Curr_KF.limage, Curr_KF.lmask, Curr_KF.lKeyPoints, Curr_KF.lDescriptors);
            // ORBfeatureAndDescriptor(Curr_KF.rimage, Curr_KF.rmask, Curr_KF.rKeyPoints, Curr_KF.rDescriptors);
            
            // Curr_KF.lkeypoint = Converter::KeyPoint2Point2f(Curr_KF.lKeyPoints);
            // Curr_KF.l_descriptors = Converter::MatToVectorMat(Curr_KF.lDescriptors);
            
            cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, true);
            // cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
            std::vector<cv::DMatch> DescriptorMatch;
            matcher->match(Curr_KF.lDescriptors, Curr_KF.rDescriptors, DescriptorMatch);
            std::sort(DescriptorMatch.begin(), DescriptorMatch.end());            
            std::vector<cv::DMatch> goodDescriptorMatch(DescriptorMatch.begin(), DescriptorMatch.begin() + DescriptorMatch.size());
            
            
            std::cout << "feature Num  : " << goodDescriptorMatch.size() << std::endl;
            Curr_KF.lkeypoint.resize(goodDescriptorMatch.size());
            Curr_KF.rkeypoint.resize(goodDescriptorMatch.size());
            Curr_KF.l_descriptors.resize(goodDescriptorMatch.size());
            Curr_KF.r_descriptors.resize(goodDescriptorMatch.size());
            for(int i = 0; i < goodDescriptorMatch.size(); i++){
                Curr_KF.lkeypoint[i] = Curr_KF.lKeyPoints[goodDescriptorMatch[i].queryIdx].pt;
                Curr_KF.rkeypoint[i] = Curr_KF.rKeyPoints[goodDescriptorMatch[i].trainIdx].pt;
                Curr_KF.l_descriptors[i] = Curr_KF.lDescriptors.row(goodDescriptorMatch[i].queryIdx);
                Curr_KF.r_descriptors[i] = Curr_KF.rDescriptors.row(goodDescriptorMatch[i].trainIdx);
            }
            
            // cv::goodFeaturesToTrack(Curr_KF.limage, Curr_KF.lkeypoint, max_keypoint, 0.01, minDis);
            // std::cout << "feature num : " << Curr_KF.lkeypoint.size() << std::endl;
            // // OpticalFlowStereo(Curr_KF.limage, Curr_KF.rimage, Curr_KF.lkeypoint, Curr_KF.rkeypoint, Curr_KF.l_descriptors);
            // OpticalFlowStereo(Curr_KF.limage, Curr_KF.rimage, Curr_KF.lkeypoint, Curr_KF.rkeypoint);
            // std::cout << "track feature num : " << Curr_KF.rkeypoint.size() << std::endl;
            
            // // Compute Descripor
            // Curr_KF.lKeyPoints = Converter::Point2f2KeyPoint(Curr_KF.lkeypoint);
            // Curr_KF.rKeyPoints = Converter::Point2f2KeyPoint(Curr_KF.rkeypoint);
            // sift->compute(Curr_KF.limage, Curr_KF.lKeyPoints, Curr_KF.lDescriptors);
            // sift->compute(Curr_KF.rimage, Curr_KF.rKeyPoints, Curr_KF.rDescriptors);
            // Curr_KF.l_descriptors = Converter::MatToVectorMat(Curr_KF.lDescriptors);
            // Curr_KF.r_descriptors = Converter::MatToVectorMat(Curr_KF.rDescriptors);

            // // debug
            // // cv::Mat mask___, descriii__;
            // // std::vector<cv::KeyPoint> adf;
            // // sift->detectAndCompute(image2, mask___, adf, descriii__);
            // // cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2);
            // // std::vector<cv::DMatch> asjkdfn__;
            // // matcher->match(Curr_KF.lDescriptors, descriii__, asjkdfn__);
            // // std::sort(asjkdfn__.begin(), asjkdfn__.end());
            // // std::vector<cv::DMatch> adfnjkasdf(asjkdfn__.begin(), asjkdfn__.begin() + 100);
            // // cv::Mat ddd;
            // // cv::drawMatches(Curr_KF.limage, Curr_KF.lKeyPoints, Curr_KF.rimage, adf, adfnjkasdf, ddd, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            // // cv::imshow("adf", ddd);

            // Curr_KF.l_descriptors = Converter::MatToVectorMat(Curr_KF.lDescriptors);
            // std::cout << Curr_KF.lDescriptors.size() <<" " << Curr_KF.l_descriptors.size() << std::endl;
            // std::cout << Curr_KF.l_descriptors[9] << std::endl;
            // std::cout << Curr_KF.lDescriptors.row(9) << std::endl;
            
            RemoveOutlierMatch(Curr_KF.lkeypoint, Curr_KF.rkeypoint, Curr_KF.l_descriptors, Curr_KF.r_descriptors);
            std::cout << "after remove outlier stereo sift(orb) 2d match : " << Curr_KF.lkeypoint.size() <<  " " << Curr_KF.rkeypoint.size() << " " << Curr_KF.l_descriptors.size() << std::endl;
            
            // draw left and right match image
            // cv::Mat KFStereoMatchImg = DrawKLTmatchLine(Curr_KF.limage, Curr_KF.rimage, Curr_KF.lkeypoint, Curr_KF.rkeypoint);
            // cv::imshow("KFStereoMatchImg", KFStereoMatchImg);
            // cv::Mat KFStereoMatchImg_ = DrawKLTmatchLine_vertical(Curr_KF.limage, Curr_KF.rimage, Curr_KF.lkeypoint, Curr_KF.rkeypoint);
            // cv::imshow("KFStereoMatchImg_vertical", KFStereoMatchImg_);


////////////////////////////////////////////////


            // // Triangulation
            cv::Mat P0 = K * Vec6To34ProjMat(KFgtPoses[KFcnt]);

            cv::Mat P1 =  K * rVec6To34ProjMat(KFgtPoses[KFcnt]);
            cv::Mat X;
            cv::triangulatePoints(P0, P1, Curr_KF.lkeypoint, Curr_KF.rkeypoint, X);
            std::vector<cv::Point3d> MapPts = ToXYZ(X);
            std::cout << " MP num before remove outlier : " << MapPts.size() << std::endl;
            RemoveMPoutlier(MapPts, Curr_KF.lkeypoint, Curr_KF.rkeypoint, Curr_KF.l_descriptors, Curr_KF.r_descriptors, KFgtPoses[KFcnt]);
            std::cout << " MP num after remove outlier : " << MapPts.size() << std::endl;
std::cout << "cam0 - cam1 extrinsic : << " << GetCam1ToCam0(Cam0ToBodyData, Cam1ToBodyData) << std::endl;
            std::cout << (P0) << std::endl;
            // std::cout << GetCam1ToCam0(Cam0ToBodyData, Cam1ToBodyData) << std::endl;
            std::cout << "p1" << (P1) << std::endl;
            // for(int i = 0; i < MapPts.size(); i++) std::cout << MapPts[i] << std::endl;


            // draw left and right match image
            cv::Mat KFStereoMatchImg = DrawKLTmatchLine(Curr_KF.limage, Curr_KF.rimage, Curr_KF.lkeypoint, Curr_KF.rkeypoint);
            cv::imshow("KFStereoMatchImg", KFStereoMatchImg);

            // Save id and MapDB
            for(int i = maxIds; i < maxIds + Curr_KF.lkeypoint.size(); i++){
                Curr_KF.lptsIds.push_back(i);
                MapDB.Map3dpts[i] = (MapPts[i - maxIds]);
                MapDB.MapDesctriptors[i] = Curr_KF.l_descriptors[i - maxIds];
            }
            
            // Extract ORB feature for dboW2
            cv::Mat mask, descrips;
            std::vector<cv::KeyPoint> keypoints_;
            // orb->detectAndCompute(Curr_KF.limage, mask, keypoints_, descrips);
            ORBfeatureAndDescriptor(Curr_KF.limage, mask, keypoints_, descrips);
            Curr_KF.descriptors = descrips.clone();           

            
            
            Curr_KF.KFid = KFcnt;
            Curr_KF.camPose = KFgtPoses[KFcnt];
            Keyframe copy_KF = Curr_KF;
            KFdb.push_back(copy_KF);
        
        ceres::Problem init_BA; 
        for(int j = 0; j < KFdb.size(); j++){
            
            Vector6d camProj = ToProjection(KFgtPoses[KFcnt]);

            
            for ( int i = 0; i < KFdb[j].lkeypoint.size(); i++){
                
                ceres::CostFunction* map_only_cost_func = map_point_only_ReprojectionError::create(KFdb[j].lkeypoint[i], camProj, fx, cv::Point2d(cx, cy));
                int id = KFdb[j].lptsIds[i];
                double* X = (double*)(&(MapDB.Map3dpts[id]));
                init_BA.AddResidualBlock(map_only_cost_func, new ceres::CauchyLoss(lossfuncParameter), X); 
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.num_threads = 8;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        std::cout << " Start optimize map point " << std::endl;
        ceres::Solve(options, &init_BA, &summary);
        std::cout << summary.BriefReport() << std::endl;                
        std::cout << " End optimize map point " << std::endl;



            maxIds += Curr_KF.lkeypoint.size();
            std::cout << Curr_KF.lptsIds.size() << std::endl;

            // prepare track
            std::cout << "prepare track" << std::endl;
            Track.EraseData();
            Track.SetLastData(Curr_KF.lkeypoint, Curr_KF.lptsIds, image1);
            
            KFcnt++;
            
            // Visualize
            glClear(GL_COLOR_BUFFER_BIT);
            showMP(MapDB.Map3dpts);
            glFlush();
            cv::waitKey();
            continue;
        }


        // // Tracking
        currTrackFrameNum++;
        Track.SetCurrImg(image1);
        OpticalFlowTracking(Track.last_image, Track.curr_image, Track.last_trackingPts, Track.curr_trackingPts, Track.trackIds);
        int afterTrackNum = Track.curr_trackingPts.size();
        std::cout << " Tracking ... " << afterTrackNum << "  " << Track.last_trackingPts.size() << std::endl;
        Track.trackingRatio = (double)afterTrackNum / (double)Track.beforetrackNum;
        
        cv::Mat MatchImg = DrawKLTmatchLine(Track.last_image, Track.curr_image, Track.last_trackingPts, Track.curr_trackingPts);
        cv::imshow("TrackingImg", MatchImg);
        

        Track.PrepareNextFrame();
        std::cout << "tracking ratio : " << Track.trackingRatio << std::endl;
        std::cout << "Curr track frame Num : " << currTrackFrameNum << std::endl;
        // cv::waitKey();
        
        // New Keyframe
        if(Track.trackingRatio < KeyframeTrackRatio || afterTrackNum < trackPtsNum || currTrackFrameNum > trackFrameNum){
            
            std::cout << " New Keyframe !! " << std::endl;
            Curr_KF.EraseClass();
            Curr_KF.limage = image1.clone();
            Curr_KF.rimage = image2.clone();
            Curr_KF.timeStamp = timestamps[KFcnt];
            
           // test sift
            sift->detectAndCompute(image1, Curr_KF.lmask, Curr_KF.lKeyPoints, Curr_KF.lDescriptors);
            sift->detectAndCompute(image2, Curr_KF.rmask, Curr_KF.rKeyPoints, Curr_KF.rDescriptors);
            // ORBfeatureAndDescriptor(image1, Curr_KF.lmask, Curr_KF.lKeyPoints, Curr_KF.lDescriptors);
            // ORBfeatureAndDescriptor(image2, Curr_KF.rmask, Curr_KF.rKeyPoints, Curr_KF.rDescriptors);

            // Curr_KF.lkeypoint = Converter::KeyPoint2Point2f(Curr_KF.lKeyPoints);
            // Curr_KF.l_descriptors = Converter::MatToVectorMat(Curr_KF.lDescriptors);

            cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, true);
            // cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
            std::vector<cv::DMatch> DescriptorMatch;
            matcher->match(Curr_KF.lDescriptors, Curr_KF.rDescriptors, DescriptorMatch);
            std::cout << DescriptorMatch.size() << std::endl;
            std::sort(DescriptorMatch.begin(), DescriptorMatch.end());            
            std::vector<cv::DMatch> goodDescriptorMatch(DescriptorMatch.begin(), DescriptorMatch.begin() + DescriptorMatch.size());
            
            //
            std::cout << "feature Num  : " << goodDescriptorMatch.size() << std::endl;
            Curr_KF.lkeypoint.resize(goodDescriptorMatch.size());
            Curr_KF.rkeypoint.resize(goodDescriptorMatch.size());
            Curr_KF.l_descriptors.resize(goodDescriptorMatch.size());
            Curr_KF.r_descriptors.resize(goodDescriptorMatch.size());
            for(int i = 0; i < goodDescriptorMatch.size(); i++){
                Curr_KF.lkeypoint[i] = Curr_KF.lKeyPoints[goodDescriptorMatch[i].queryIdx].pt;
                Curr_KF.rkeypoint[i] = Curr_KF.rKeyPoints[goodDescriptorMatch[i].trainIdx].pt;
                Curr_KF.l_descriptors[i] = Curr_KF.lDescriptors.row(goodDescriptorMatch[i].queryIdx);
                Curr_KF.r_descriptors[i] = Curr_KF.rDescriptors.row(goodDescriptorMatch[i].trainIdx);

            }
            
            // cv::goodFeaturesToTrack(Curr_KF.limage, Curr_KF.lkeypoint, max_keypoint, 0.01, minDis);
            // std::cout << "feature num : " << Curr_KF.lkeypoint.size() << std::endl;
            // // OpticalFlowStereo(Curr_KF.limage, Curr_KF.rimage, Curr_KF.lkeypoint, Curr_KF.rkeypoint, Curr_KF.l_descriptors);
            // OpticalFlowStereo(Curr_KF.limage, Curr_KF.rimage, Curr_KF.lkeypoint, Curr_KF.rkeypoint);
            // std::cout << "track feature num : " << Curr_KF.rkeypoint.size() << std::endl;
            
            // // Compute Descripor
            // Curr_KF.lKeyPoints = Converter::Point2f2KeyPoint(Curr_KF.lkeypoint);
            // Curr_KF.rKeyPoints = Converter::Point2f2KeyPoint(Curr_KF.rkeypoint);
            // sift->compute(Curr_KF.limage, Curr_KF.lKeyPoints, Curr_KF.lDescriptors);
            // sift->compute(Curr_KF.rimage, Curr_KF.rKeyPoints, Curr_KF.rDescriptors);
            // Curr_KF.l_descriptors = Converter::MatToVectorMat(Curr_KF.lDescriptors);
            // Curr_KF.r_descriptors = Converter::MatToVectorMat(Curr_KF.rDescriptors);
            
            //  // // Compute Descripor
            // // Curr_KF.lKeyPoints = Converter::Point2f2KeyPoint(Curr_KF.lkeypoint);
            // Curr_KF.rKeyPoints = Converter::Point2f2KeyPoint(Curr_KF.rkeypoint);
            // // sift->compute(Curr_KF.limage, Curr_KF.lKeyPoints, Curr_KF.lDescriptors);
            // sift->compute(Curr_KF.rimage, Curr_KF.rKeyPoints, Curr_KF.rDescriptors);
            // Curr_KF.r_descriptors = Converter::MatToVectorMat(Curr_KF.rDescriptors);           
            
            
            RemoveOutlierMatch(Curr_KF.lkeypoint, Curr_KF.rkeypoint, Curr_KF.l_descriptors, Curr_KF.r_descriptors);
            std::cout << "after remove outlier sift match : " << Curr_KF.lkeypoint.size() << std::endl;


            // cv::Mat KFStereoMatchImg_ = DrawKLTmatchLine_vertical(Curr_KF.limage, Curr_KF.rimage, Curr_KF.lkeypoint, Curr_KF.rkeypoint);
            // cv::imshow("KFStereoMatchImg_vertical", KFStereoMatchImg_);

            // // Triangulation
            cv::Mat P0 = K * Vec6To34ProjMat(KFgtPoses[KFcnt]);
            cv::Mat P1 = K * rVec6To34ProjMat(KFgtPoses[KFcnt]);
            std::cout << Proj34ToPose(P0) << std::endl;
            std::cout << GetCam1ToCam0(Cam0ToBodyData, Cam1ToBodyData) << std::endl;
            std::cout << Proj34ToPose(P1) << std::endl;
            std::cout << (P0) << std::endl;
            std::cout << (P1) << std::endl;
            cv::Mat X;
            cv::triangulatePoints(P0, P1, Curr_KF.lkeypoint, Curr_KF.rkeypoint, X);
            std::vector<cv::Point3d> MapPts_ = ToXYZ(X);
            std::cout << "MP num : " << MapPts_.size() << std::endl;
            // for(auto i : MapPts_) std::cout << i << std::endl;
            RemoveMPoutlier(MapPts_, Curr_KF.lkeypoint, Curr_KF.rkeypoint, Curr_KF.l_descriptors, Curr_KF.r_descriptors, KFgtPoses[KFcnt]);
            std::cout << "after remove MP outlier MP num : " << MapPts_.size() << std::endl;

            // draw left and right match image
            cv::Mat KFStereoMatchImg = DrawKLTmatchLine(Curr_KF.limage, Curr_KF.rimage, Curr_KF.lkeypoint, Curr_KF.rkeypoint);
            cv::imshow("KFStereoMatchImg", KFStereoMatchImg);
            cv::Mat KFStereoMatchImg_ = DrawKLTmatchLine_vertical(Curr_KF.limage, Curr_KF.rimage, Curr_KF.lkeypoint, Curr_KF.rkeypoint);
            cv::imshow("KFStereoMatchImg_vertical", KFStereoMatchImg_);

            // Save id and MapDB
            // const int MaxId = maxIds;
            int newPtsNum(0), samePtsNum(0);
            for(int i = 0; i < Curr_KF.lkeypoint.size(); i++){
                
                bool samePts = false;
                double min_diff= 1000;
                int min_idx = -1;
                for(int j = 0; j < Track.curr_trackingPts.size(); j++){
                    
                    double diff_x = std::fabs(Curr_KF.lkeypoint[i].x - Track.curr_trackingPts[j].x);
                    double diff_y = std::fabs(Curr_KF.lkeypoint[i].y - Track.curr_trackingPts[j].y);
                    double diff = std::sqrt(diff_x * diff_x + diff_y * diff_y);
                    if(diff < min_diff){
                        min_diff = diff; 
                        min_idx = j;
                    }
                }

                if(min_diff < 1.0 ) samePts = true;
                if(samePts){    
                    
                        // std::cout << "Find same point " << std::endl;
                        int idpts = Track.trackIds[min_idx];
                        Curr_KF.lptsIds.push_back(idpts);
                        Track.trackIds.erase(Track.trackIds.begin() + min_idx);
                        Track.curr_trackingPts.erase(Track.curr_trackingPts.begin() + min_idx);
                        cv::Mat kajd = (Curr_KF.l_descriptors[i] + MapDB.MapDesctriptors[idpts]) / 2;
                        MapDB.MapDesctriptors[idpts].release();
                        MapDB.MapDesctriptors[idpts] = kajd;
                        samePtsNum++;
                }
                if(!samePts){

                    // std::cout << " New point " << std::endl;
                    Curr_KF.lptsIds.push_back(maxIds);
                    MapDB.Map3dpts[maxIds] = MapPts_[i];
                    MapDB.MapDesctriptors[maxIds] = Curr_KF.l_descriptors[i];
                    // MapDB.MapIds.push_back(maxIds);
                    maxIds++;
                    newPtsNum++;

                }
            }
            
            // Extract ORB feature for dboW2
            cv::Mat mask, descrips;
            std::vector<cv::KeyPoint> keypoints_;
            // orb->detectAndCompute(Curr_KF.limage, mask, keypoints_, descrips);
            ORBfeatureAndDescriptor(Curr_KF.limage, mask, keypoints_, descrips);
            Curr_KF.descriptors = descrips.clone();             

            
            Curr_KF.camPose = KFgtPoses[KFcnt];
            Curr_KF.KFid = KFcnt;
            Keyframe copy_KF = Curr_KF;
            KFdb.push_back(copy_KF);

            // Optimize !!
        ceres::Problem globalBA; 
        std::cout << "kf size : " <<  KFdb.size() << std::endl;
        for(int j = 0; j < KFdb.size(); j++){
            
            Vector6d camProj = ToProjection(KFdb[j].camPose);

            
            for ( int i = 0; i < KFdb[j].lkeypoint.size(); i++){
                
                ceres::CostFunction* map_only_cost_func = map_point_only_ReprojectionError::create(KFdb[j].lkeypoint[i], camProj, fx, cv::Point2d(cx, cy));
                int id = KFdb[j].lptsIds[i];
                double* X = (double*)(&(MapDB.Map3dpts[id]));
                globalBA.AddResidualBlock(map_only_cost_func, NULL, X); 
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.max_num_iterations = 1000;
        options.num_threads = 8;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        std::cout << " Start optimize map point " << std::endl;
        ceres::Solve(options, &globalBA, &summary);
        std::cout << summary.BriefReport() << std::endl;                
        std::cout << " End optimize map point " << std::endl;           




            std::cout << Curr_KF.lkeypoint.size() << " " << Curr_KF.lptsIds.size() << std::endl;
            std::cout << "new num : " << newPtsNum << "    same num : " << samePtsNum << std::endl;
            std::cout << " total landmark Num : " << MapDB.Map3dpts.size() << std::endl;
            std::cout << " Total Keyframe Num  : " << KFdb.size() << std::endl;



            // prepare track
            std::cout << "prepare track" << std::endl;
            Track.EraseData();
            Track.SetLastData(Curr_KF.lkeypoint, Curr_KF.lptsIds, image1);
            currTrackFrameNum = 0;
            //  cv::waitKey();
        }
        
        // Visualize
        glClear(GL_COLOR_BUFFER_BIT);
        showMP(MapDB.Map3dpts);
        for(int i = 0; i < KFdb.size(); i++){
            int idx = KFdb[i].KFid;
            Eigen::Matrix4d Pose44 = To44RT(KFgtPoses[idx]);
            show_trajectory_keyframe(Pose44, 0.0, 0.0, 1.0, 0.1, 0.2);
        }
        glFlush();
        cv::imshow("current left view", Curr_KF.limage);
        cv::waitKey(10);
        KFcnt++;
    // }
}
s.close();

/////////////////////////////////  Save result
    std::cout << "Save result to bin file ... " << std::endl;
    DataBase *db;
    db = new DataBase();
    db->SaveResultToDB(&MapDB, &KFdb);
    
    std::string saveDbPath = "EurocMH01_DB_original.bin";
    std::ofstream out(saveDbPath, std::ios_base::binary);
    if (!out)
    {
        std::cout << "Cannot Write to Database File: " << std::endl;
        exit(-1);
    }
    boost::archive::binary_oarchive oa(out, boost::archive::no_header);
    oa << db;
    cv::waitKey();
    return 0;
}








        