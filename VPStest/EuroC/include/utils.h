#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"

#include "Parameter.h"






extern Eigen::Vector3d Origin;
extern Eigen::Vector3d ZVec;
extern Eigen::Matrix3d Iden;

//////////// EuroC ////////////////////

extern float fx, fy, cx, cy;
extern float IntrinsicData[];
      
extern double Cam2BodyData[];

cv::Mat GetK(float* IntrinsicData);
Eigen::Matrix4d GetCam2Body(double * Cam2BodyData);
std::vector<Eigen::Vector3d> Mat3XdToVec3d(Eigen::Matrix3Xd LidarPoints);
Eigen::Vector3d ToVec3(Eigen::Matrix3d rot);
Eigen::Vector3f ToVec3(Eigen::Matrix3f rot);
Eigen::Matrix3d ToMat33(Eigen::Vector3d rod);
Eigen::Matrix3f ToMat33(Eigen::Vector3f rod);
Vector6d To6DOF(Eigen::Quaterniond q, Eigen::Vector3d t);
Vector6d To6DOF(Eigen::Matrix4d RT);
Vector6d To6DOF(cv::Mat R, cv::Mat T);
Eigen::Quaterniond ToQuaternion(const Vector6d Pose);
Eigen::Matrix4f To44RT(Vector6f pose);
Eigen::Matrix4d To44RT(Vector6d pose);
Eigen::Matrix4d To44RT(std::vector<double> pose);
Vector6d ProjectionTo6DOFPoses(cv::Mat R, cv::Mat T);
Vector6d ProjectionTo6DOFPoses(Vector6d proj);
double ToAngle(Eigen::Matrix4d LidarRotation);
Eigen::Vector3d ToAxis(Eigen::Matrix4d LidarRotation);
float VerticalAngle(Eigen::Vector3d p);
double PointDistance(Eigen::Vector3d p);
double PointDistance(Eigen::Vector3d p1, Eigen::Vector3d p2);
double CosRaw2(double a, double b, float ang);
double Rad2Degree(double rad);
double Ddegree2Rad(double degree);

// namespace constants
// {
//     // cv::Mat K;


//     const int abc = 1;
// }