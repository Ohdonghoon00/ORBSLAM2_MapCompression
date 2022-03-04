#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <map>


class Keyframe
{

public:
    
    Keyframe()
    {}

    // std::map< int, std::vector<cv::KeyPoint> > KeyPoints;
    // std::map< int, cv::Mat > masks;
    // std::map< int, cv::Mat > Descriptors;

    std::vector<cv::Point2f> keypoint;
    std::vector<cv::KeyPoint> KeyPoints;
    cv::Mat mask;
    cv::Mat Descriptors;

    Keyframe(const Keyframe &tc)
    {
        keypoint = tc.keypoint;
        KeyPoints = tc.KeyPoints;
        mask = tc.mask;
        Descriptors = tc.Descriptors;
    }

    void EraseClass()
    {
        keypoint.clear();
        std::vector<cv::Point2f>().swap(keypoint);
        KeyPoints.clear();
        std::vector<cv::KeyPoint>().swap(KeyPoints);
        mask.release();
        Descriptors.release();
    }
};