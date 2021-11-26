#pragma once

#include<opencv2/core.hpp>
#include "opencv2/opencv.hpp"












// Converter
cv::Mat VectorMatToMat(std::vector<cv::Mat> Descriptor)
{
    cv::Mat Descriptors(Descriptor[0]);
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

Eigen::Quaternionf ToQuaternion(Eigen::Matrix4f &pose)
{
    Eigen::Matrix3f rot;
    rot <<  pose(0, 0), pose(0, 1), pose(0, 2),
            pose(1, 0), pose(1, 1), pose(1, 2),
            pose(2, 0), pose(2, 1), pose(2, 2);

    Eigen::Quaternionf q(rot);

    return q;
}