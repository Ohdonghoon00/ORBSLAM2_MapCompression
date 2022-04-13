#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"

#include "Converter.h"
#include "Parameter.h"



// typedef Eigen::Matrix<float, 6, 1> Vector6f;
// typedef Eigen::Matrix<double, 6, 1> Vector6d;


extern Eigen::Vector3d Origin;
extern Eigen::Vector3d ZVec;
extern Eigen::Matrix3d Iden;

//////////// EuroC ////////////////////

extern double fx, fy, cx, cy;
extern double IntrinsicData[];
      
extern double Cam0ToBodyData[];
extern double Cam1ToBodyData[];
extern cv::Point2d c;


cv::Mat GetK(double* IntrinsicData);
cv::Mat GetKf(float* IntrinsicData);
Eigen::Matrix4d GetCam2Body(double * Cam2BodyData);
Eigen::Matrix4d GetCam1ToCam0(double * Cam2BodyData0, double * Cam2BodyData1);
int ReadgtPose(const std::string gtpath, std::vector<Vector6d>* poses);
std::vector<Eigen::Vector3d> Mat3XdToVec3d(Eigen::Matrix3Xd LidarPoints);
Eigen::Vector3d ToVec3(Eigen::Matrix3d rot);
Eigen::Vector3f ToVec3(Eigen::Matrix3f rot);
Eigen::Matrix3d ToMat33(Eigen::Vector3d rod);
Eigen::Matrix3f ToMat33(Eigen::Vector3f rod);
Vector6d To6DOF(Eigen::Quaterniond q, Eigen::Vector3d t);
Vector6d To6DOF(Eigen::Matrix4d RT);
Eigen::Quaterniond ToQuaternion(const Vector6d Pose);
Eigen::Matrix4f To44RT(Vector6f pose);
Eigen::Matrix4d To44RT(Vector6d pose);
Eigen::Matrix4d To44RT(std::vector<double> pose);
cv::Mat Vec6To34ProjMat(Vector6d pose);
cv::Mat rVec6To34ProjMat(Vector6d pose);
Eigen::Matrix4d Proj34ToPose(cv::Mat Proj);
double ToAngle(Eigen::Matrix4d LidarRotation);
Eigen::Vector3d ToAxis(Eigen::Matrix4d LidarRotation);
Vector6d ToProjection(Vector6d pose);
float VerticalAngle(Eigen::Vector3d p);
double PointDistance(Eigen::Vector3d p);
double PointDistance(Eigen::Vector3d p1, Eigen::Vector3d p2);
double CosRaw2(double a, double b, float ang);
double Rad2Degree(double rad);
double Ddegree2Rad(double degree);
std::vector<cv::Point3d> ToXYZ(cv::Mat &X);
std::vector<float> ReprojectionError(std::vector<cv::Point3d> WPts, std::vector<cv::Point2f> ImgPts, Eigen::Matrix4d Pose);
int FindTimestampIdx(const double a, const std::vector<double> b);
int readCsvGtPose(std::string gtpath, std::vector<Vector6d>* poses, std::vector<double>* timeStamps);
void OpticalFlowStereo(cv::Mat previous, cv::Mat current, std::vector<cv::Point2f> &previous_pts, std::vector<cv::Point2f> &current_pts);
void OpticalFlowTracking(cv::Mat previous, cv::Mat current, std::vector<cv::Point2f> &previous_pts, std::vector<cv::Point2f> &current_pts, std::vector<int> &trackIds);
cv::Mat DrawKLTmatchLine(cv::Mat image1, cv::Mat image2, std::vector<cv::Point2f> previous_pts, std::vector<cv::Point2f> current_pts);
cv::Mat DrawKLTmatchLine_vertical(cv::Mat image1, cv::Mat image2, std::vector<cv::Point2f> previous_pts, std::vector<cv::Point2f> current_pts);

void RemoveMPoutlier(std::vector<cv::Point3d> &mp, std::vector<cv::Point2f> &lpts, std::vector<cv::Point2f> &rpts, const Vector6d pose);
void RemoveOutlierMatch(std::vector<cv::Point2f> &lpts, std::vector<cv::Point2f> &rpts);


// namespace constants
// {
//     // cv::Mat K;


//     const int abc = 1;
// }