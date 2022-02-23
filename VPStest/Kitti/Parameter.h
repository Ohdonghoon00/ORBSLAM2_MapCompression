#pragma once

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"    
#include <Eigen/Dense>


typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<float, 6, 1> Vector6f;
    
// Camera Parameter

//////////// Kitti //////////////////
// 00-02
// float fx(718.856), fy(718.856), cx(607.193), cy(185.216);
// float data[] = {fx, 0.0,cx, 
//                 0.0, fy, cy,
//                 0.0, 0.0, 1.0};       
// cv::Mat K(3, 3, CV_32F, data);

// 03
// float fx(721.5377), fy(721.5377), cx(609.5593), cy(172.8540);
// float data[] = {fx, 0.0,cx, 
//                 0.0, fy, cy,
//                 0.0, 0.0, 1.0};       
// cv::Mat K(3, 3, CV_32F, data);

// 04-12
// float fx(707.0912), fy(707.0912), cx(601.8873), cy(183.1104);
// float data[] = {fx, 0.0,cx, 
//                 0.0, fy, cy,
//                 0.0, 0.0, 1.0};       
// cv::Mat K(3, 3, CV_32F, data);


//////////// EuroC ////////////////////

float fx(435.2046959714599), fy(435.2046959714599), cx(367.4517211914062), cy(252.2008514404297);
float data[] = {fx, 0.0,cx, 
                0.0, fy, cy,
                0.0, 0.0, 1.0};       
cv::Mat K(3, 3, CV_32F, data);