#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <map>
#include "Parameter.h"


class Keyframe
{

public:
    
    Keyframe()
    {}


    cv::Mat limage, rimage;
    int KFid;
    double timeStamp;
    Vector6d camPose;
    std::vector<cv::Point2f> lkeypoint, rkeypoint;
    std::vector<int> lptsIds;
    
    cv::Mat lmask, rmask;
    cv::Mat lDescriptors, rDescriptors;
    std::vector<cv::Mat> l_descriptors, r_descriptors;
    std::vector<cv::KeyPoint> lKeyPoints, rKeyPoints;

    // for dbow2
    cv::Mat descriptors;

    Keyframe(const Keyframe &tc)
    {
        lkeypoint = tc.lkeypoint;
        rkeypoint = tc.rkeypoint;
        lptsIds = tc.lptsIds;
        
        KFid = tc.KFid;
        camPose = tc.camPose;
        
        lKeyPoints = tc.lKeyPoints;
        rKeyPoints = tc.rKeyPoints;
        lmask = tc.lmask;
        rmask = tc.rmask;
        lDescriptors = tc.lDescriptors;
        rDescriptors = tc.rDescriptors;
        l_descriptors = tc.l_descriptors;
        r_descriptors = tc.r_descriptors;

        limage = tc.limage;
        rimage = tc.rimage;

        descriptors = tc.descriptors;
    }

    void EraseClass()
    {
        lkeypoint.clear();
        rkeypoint.clear();
        lptsIds.clear();
        
        rKeyPoints.clear();
        lKeyPoints.clear();
        lmask.release();
        rmask.release();
        lDescriptors.release();
        rDescriptors.release();
        l_descriptors.clear();
        r_descriptors.clear();
        limage.release();
        rimage.release();
        descriptors.release();
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
        last_trackingPts.clear();
        trackIds.clear();   
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