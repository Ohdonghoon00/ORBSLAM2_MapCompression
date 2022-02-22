#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"

typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

float fx(435.2046959714599), fy(435.2046959714599), cx(367.4517211914062), cy(252.2008514404297);
float data[] = {fx, 0.0,cx, 
                0.0, fy, cy,
                0.0, 0.0, 1.0};       
cv::Mat K(3, 3, CV_32F, data);

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
double ToAngle(Eigen::Matrix4d LidarRotation);
Eigen::Vector3d ToAxis(Eigen::Matrix4d LidarRotation);
float VerticalAngle(Eigen::Vector3d p);
double PointDistance(Eigen::Vector3d p);
double PointDistance(Eigen::Vector3d p1, Eigen::Vector3d p2);
double CosRaw2(double a, double b, float ang);

namespace constants
{
    // cv::Mat K;
    extern const Eigen::Matrix4d Cam2Body;
}