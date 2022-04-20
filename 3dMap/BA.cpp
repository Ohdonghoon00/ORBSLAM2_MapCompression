#include <iostream>
#include<algorithm>
#include<fstream>

#include "BoostArchiver.h"
#include "DataBase.h"
#include "BA.h"
#include "utils.h"
#include "map_viewer.h"
#include "ORBextractor.h"

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

using ceres::CauchyLoss;
using ceres::HuberLoss;
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{

    DataBase* DB;

    // Load Map
    std::ifstream in(argv[1], std::ios_base::binary);
    if (!in){
        std::cout << "Cannot DataBase bin file is empty!" << std::endl;
        return false;
    }
    boost::archive::binary_iarchive ia(in, boost::archive::no_header);
    ia >> DB;
    in.close();
    std::cout << DB->KFtoMPIdx.size() << std::endl;
    // Load Euroc GT pose
    std::string EurocGTPath = argv[2];
    std::vector<Vector6d> EurocgtPoses;
    std::vector<double> timeStamps;
    
    ReadgtPose(EurocGTPath, &EurocgtPoses, &timeStamps);
    // ReadKFPose(EurocGTPath, &EurocgtPoses, &timeStamps);
    std::cout << "EurocgtPoses size : " << EurocgtPoses.size() << std::endl;
    
    // Save DB kfPose and descriptor
    int nFeatures = 6000;
    float scaleFactor = 1.2;
    int nlevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;
    double lossfuncParameter = 1 / ((fx + fy) / 2);
    ORBextractor ORBfeatureAndDescriptor(nFeatures, scaleFactor, nlevels, iniThFAST, minThFAST);    

    // int idx = 300;
    int outlierNum = 0;
    for(size_t i = 0; i < DB->KFtoMPIdx.size(); i++){

        size_t imgIdx = i;
        
        int poseidx = FindTimestampIdx(DB->timestamps[imgIdx], timeStamps);
        std::vector<cv::Point3d> Landmarks = DB->GetKF3dPoint(imgIdx);
        // std::vector<cv::Point3f> Landmarksf;
        // for(auto i : Landmarks)
        //     Landmarksf.push_back(cv::Point3f((float)i.x, (float)i.y, (float)i.z));
        std::vector<cv::KeyPoint> imgPoints_ = DB->GetKF2dPoint(imgIdx);
        std::vector<cv::Point2f> imgPoints = Converter::KeyPoint2Point2f(imgPoints_);
        std::vector<float> ReprojErr = ReprojectionError(Landmarks, imgPoints, To44RT(EurocgtPoses[poseidx]));
        for(int j = 0; j < ReprojErr.size(); j++){
            if(ReprojErr[j] > 5.0) outlierNum++;
        }
    }
    std::cout << " outlier Num : " << outlierNum << std::endl;
    
    // Viewer
    glutInit(&argc, argv);
    initialize_window();
        
        for(int i = 0; i < DB->Landmarks.size(); i++){
            GLdouble X_map(DB->Landmarks[i].x), Y_map(DB->Landmarks[i].y), Z_map(DB->Landmarks[i].z);
            show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 0.0, 0.01);
        }
        for(int i = 0; i < DB->KFtoMPIdx.size(); i++){
            int poseIdx = FindTimestampIdx(DB->timestamps[i], timeStamps);
            // int poseIdx = i;
            Eigen::Matrix4d Pose44 = To44RT(EurocgtPoses[poseIdx]);
            show_trajectory_keyframe(Pose44, 0.0, 0.0, 1.0, 0.1, 0.2);
        }
        glFlush();
        
        ceres::Problem global_BA; 
        DB->kfPoses.clear();
        DB->dbow2Descriptors.clear();
        for(int j = 0; j < DB->KFtoMPIdx.size(); j++){
            
            std::vector<cv::KeyPoint> imgPoints_ = DB->GetKF2dPoint(j);
            std::vector<cv::Point2f> imgPoints = Converter::KeyPoint2Point2f(imgPoints_);
            
            int poseIdx = FindTimestampIdx(DB->timestamps[j], timeStamps);
            // int poseIdx = j;
            std::cout << imgPoints.size() << " ";
            Vector6d camProj = ToProjection(EurocgtPoses[poseIdx]);
            for ( int i = 0; i < imgPoints.size(); i++){
                ceres::CostFunction* map_only_cost_func = map_point_only_ReprojectionError::create(imgPoints[i], camProj, fx, cv::Point2d(cx, cy));
                int id = DB->KFtoMPIdx[j][i];
                double* X = (double*)(&(DB->Landmarks[id]));
                global_BA.AddResidualBlock(map_only_cost_func, new ceres::CauchyLoss(lossfuncParameter), X); 
            }
            cv::Mat mask, Descriptors;
            std::vector<cv::KeyPoint> keypoint;
            cv::Mat imageDB = DB->LeftKFimg[j];            
            ORBfeatureAndDescriptor(imageDB, mask, keypoint, Descriptors);
            DB->kfPoses[j] = EurocgtPoses[poseIdx];
            DB->dbow2Descriptors[j] = Descriptors;            

        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.num_threads = 8;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        std::cout << " Start optimize map point " << std::endl;
        std::cout << DB->Landmarks[1000] << std::endl;
        // ceres::Solve(options, &global_BA, &summary);
        std::cout << summary.BriefReport() << std::endl;                
        std::cout << " End optimize map point " << std::endl;
        std::cout << DB->Landmarks[1000] << std::endl;
        std::cout << fx << std::endl;
        std::cout << cx << std::endl; 
        std::cout << cy << std::endl;
        
        glClear(GL_COLOR_BUFFER_BIT);
        for(int i = 0; i < DB->Landmarks.size(); i++){
            GLdouble X_map(DB->Landmarks[i].x), Y_map(DB->Landmarks[i].y), Z_map(DB->Landmarks[i].z);
            show_trajectory(X_map, Y_map, Z_map, 0.0, 0.0, 0.0, 0.01);
        }
        for(int i = 0; i < DB->KFtoMPIdx.size(); i++){
            int poseIdx = FindTimestampIdx(DB->timestamps[i], timeStamps);
            // int poseIdx = i;
            Eigen::Matrix4d Pose44 = To44RT(EurocgtPoses[poseIdx]);
            show_trajectory_keyframe(Pose44, 0.0, 0.0, 1.0, 0.1, 0.2);
        }
        glFlush();
        
        std::string saveDbPath = "EurocMH01_DB_original_.bin";
        std::ofstream out(saveDbPath, std::ios_base::binary);
        if (!out)
        {
            std::cout << "Cannot Write to Database File: " << std::endl;
            exit(-1);
        }
        boost::archive::binary_oarchive oa(out, boost::archive::no_header);
        oa << DB;
        out.close();





    cv::imshow("ddd", DB->LeftKFimg[0]);
    cv::waitKey();
    return 0;
}