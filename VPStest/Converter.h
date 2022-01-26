#pragma once

#include<opencv2/core.hpp>
#include "opencv2/opencv.hpp"

#include "Parameter.h"











// Converter
cv::Mat VectorMatToMat(std::vector<cv::Mat> Descriptor)
{
    cv::Mat Descriptors(Descriptor.front());
    for(size_t i = 1; i < Descriptor.size(); i++){
        cv::vconcat(Descriptors, Descriptor[i], Descriptors);
    }

    return Descriptors;
}

std::vector<cv::Mat> MatToVectorMat(cv::Mat Descriptor)
{
    std::vector<cv::Mat> Descriptors;
    Descriptors.resize(Descriptor.rows);
    for(int i = 0; i < Descriptor.rows; i++){
        Descriptors[i] = Descriptor.row(i);
    }

    return Descriptors;
}

Eigen::Quaterniond ToQuaternion(Eigen::Matrix4d &pose)
{
    Eigen::Matrix3d rot;
    rot <<  pose(0, 0), pose(0, 1), pose(0, 2),
            pose(1, 0), pose(1, 1), pose(1, 2),
            pose(2, 0), pose(2, 1), pose(2, 2);

    Eigen::Quaterniond q(rot);

    return q;
}

std::vector<cv::KeyPoint> Point2f2KeyPoint(std::vector<cv::Point2f> pts2f)
{
    std::vector<cv::KeyPoint> KeyPoints;
    cv::KeyPoint::convert(pts2f, KeyPoints);

    return KeyPoints;
}

std::vector<cv::Point2f> KeyPoint2Point2f(std::vector<cv::KeyPoint> KeyPoints)
{
    std::vector<cv::Point2f> pts2f;
    cv::KeyPoint::convert(KeyPoints, pts2f);

    return pts2f;
}

Vector6d To6DOF(Eigen::Matrix4d RT)
{
    Eigen::Matrix3d R = RT.block<3, 3>(0, 0);
    Eigen::AngleAxisd rod(R);
    Eigen::Vector3d r(rod.axis());
    double angle = rod.angle();
    r *= angle;
    
    Vector6d Pose;
    Pose << r.x(), r.y(), r.z(), RT(0, 3), RT(1, 3), RT(2, 3);
    return Pose;
}

Eigen::Matrix4d To44RT(Vector6d rot)
{

    cv::Mat R( 1, 3, CV_64FC1);
    R.at<double>(0, 0) = rot[0];
    R.at<double>(0, 1) = rot[1];
    R.at<double>(0, 2) = rot[2];

    cv::Rodrigues(R, R);

    Eigen::Matrix4d RT;
    RT << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), rot[3],
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), rot[4],
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), rot[5],
                0,                 0,                   0,                  1;

    return RT;
}

Eigen::Vector3d To3DOF(Eigen::Quaterniond q)
{
    Eigen::Matrix3d R(q);
    Eigen::AngleAxisd rod(R);
    Eigen::Vector3d r(rod.axis());
    double angle = rod.angle();
    r *= angle;

    Eigen::Vector3d r_pose;
    r_pose << r.x(), r.y(), r.z();

    return r_pose;
}