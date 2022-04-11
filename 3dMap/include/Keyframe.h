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


    std::vector<cv::Point2f> lkeypoint, rkeypoint;
    std::vector<int> lptsIds;
    std::vector<cv::KeyPoint> KeyPoints;
    cv::Mat mask;
    cv::Mat Descriptors;

    Keyframe(const Keyframe &tc)
    {
        lkeypoint = tc.lkeypoint;
        rkeypoint = tc.rkeypoint;
        KeyPoints = tc.KeyPoints;
        mask = tc.mask;
        Descriptors = tc.Descriptors;
    }

    void EraseClass()
    {
        lkeypoint.clear();
        std::vector<cv::Point2f>().swap(lkeypoint);
        rkeypoint.clear();
        std::vector<cv::Point2f>().swap(rkeypoint);
        KeyPoints.clear();
        std::vector<cv::KeyPoint>().swap(KeyPoints);
        mask.release();
        Descriptors.release();
    }
};

struct Track
{
    
    Track()
    {}

    std::vector<cv::Point2f> last_trackingPts, curr_trackingPts;
    std::vector< int > trackIds;

    cv::Mat last_image, curr_image;

    double trackingRatio;
    int beforetrackNum;

    void EraseData()
    {
        curr_trackingPts.clear();
        std::vector<cv::Point2f>().swap(curr_trackingPts);
        last_trackingPts.clear();
        std::vector<cv::Point2f>().swap(last_trackingPts);
        trackIds.clear();   
        std::vector<int>().swap(trackIds);
        curr_image.release();
        last_image.release();
    }

    void SetCurrImg(cv::Mat img)
    {
        curr_image = img.clone();
    }

    void SetLastData(std::vector<cv::Point2f> &pts, std::vector<int> &ids, cv::Mat img)
    {
        last_image = img.clone();
        trackIds.assign(ids.begin(), ids.end());
        last_trackingPts.assign(pts.begin(), pts.end());
        beforetrackNum = pts.size();
    }

    void PrepareNextFrame()
    {
        last_image.release();
        last_image = curr_image.clone();
        last_trackingPts.clear();
        last_trackingPts.assign(curr_trackingPts.begin(), curr_trackingPts.end());
    }
};