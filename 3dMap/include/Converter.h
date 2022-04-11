#pragma once

#include<opencv2/core.hpp>
#include "opencv2/opencv.hpp"

#include "Parameter.h"
namespace Converter{
// Converter
cv::Mat VectorMatToMat(std::vector<cv::Mat> Descriptor);


std::vector<cv::Mat> MatToVectorMat(cv::Mat Descriptor);



std::vector<cv::KeyPoint> Point2f2KeyPoint(std::vector<cv::Point2f> pts2f);


std::vector<cv::Point2f> KeyPoint2Point2f(std::vector<cv::KeyPoint> KeyPoints);


// Eigen::Vector3d To3DOF(Eigen::Quaterniond q);


Eigen::Matrix4Xf HomogeneousForm(std::vector<cv::Point3f> Wpts);


Eigen::Matrix3Xf HomogeneousForm(std::vector<cv::Point2f> Imgpts);


Eigen::MatrixXf Mat2Eigen(cv::Mat a);

}