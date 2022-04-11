#include "Converter.h"



namespace Converter{

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

// Eigen::Vector3d To3DOF(Eigen::Quaterniond q)
// {
//     Eigen::Matrix3d R(q);
//     Eigen::AngleAxisd rod(R);
//     Eigen::Vector3d r(rod.axis());
//     double angle = rod.angle();
//     r *= angle;

//     Eigen::Vector3d r_pose;
//     r_pose << r.x(), r.y(), r.z();

//     return r_pose;
// }

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
            b(i, j) = a.at<float>(i, j);
        }
    }
    return b;
}
}
