#pragma once

#include<System.h>
#include<opencv2/core/core.hpp>
#include "DataBase.h"

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

cv::Mat InputQueryImg(std::string QueryFile);
std::vector<cv::KeyPoint> ORBFeatureExtract(cv::Mat img);
cv::Mat ORBDescriptor(cv::Mat img, std::vector<cv::KeyPoint> keypoints);
std::vector<cv::DMatch> ORBDescriptorMatch(cv::Mat trainDescriptor, cv::Mat queryDescriptor);

// Converter
cv::Mat VectorMatToMat(std::vector<cv::Mat> Descriptor);
std::vector<cv::Mat> MatToVectorMat(cv::Mat Descriptor);

std::vector<cv::Point3f> Sort3dpointByMatch(std::vector<cv::Point3f> Disorder3dpoint, std::vector<cv::DMatch> matches);
