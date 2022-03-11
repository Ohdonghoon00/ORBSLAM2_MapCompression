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

Eigen::Matrix4Xf HomogeneousForm(std::vector<cv::Point3f> Wpts)
{
    Eigen::Matrix4Xf WPts(4, Wpts.size());
    for(int i = 0; i < Wpts.size(); i++){
        WPts(0, i) = Wpts[i].x;
        WPts(1, i) = Wpts[i].y;
        WPts(2, i) = Wpts[i].z;
        WPts(3, i) = 1.0f;
    }
    return WPts;   
}

Eigen::Matrix3Xf HomogeneousForm(std::vector<cv::Point2f> Imgpts)
{
    Eigen::Matrix3Xf ImgPts(3, Imgpts.size());
    for(int i = 0; i < Imgpts.size(); i++){
        ImgPts(0, i) = Imgpts[i].x;
        ImgPts(1, i) = Imgpts[i].y;
        ImgPts(2, i) = 1.0f;
    }
    return ImgPts;   
}

Eigen::MatrixXf Mat2Eigen(cv::Mat a)
{
    Eigen::MatrixXf b(a.rows, a.cols);
    for(int i = 0; i < b.rows(); i++){
        for(int j = 0; j < b.cols(); j++){
            b(i, j) = (float)a.at<double>(i, j);
        }
    }
    return b;
}