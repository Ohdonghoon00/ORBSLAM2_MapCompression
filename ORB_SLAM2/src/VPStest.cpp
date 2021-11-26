#include "VPStest.h"


cv::Mat InputQueryImg(std::string QueryFile)
{ 
    cv::Mat image;
    cv::VideoCapture video;
    if(!video.open(QueryFile)){
        std::cout << " No query image " << std::endl;
        cv::waitKey();
    }
    video >> image;

    return image;
}

std::vector<cv::KeyPoint> ORBFeatureExtract(cv::Mat img)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(4000);
    vector<cv::KeyPoint> keypoints;
    orb->detect(img, keypoints);
    
    return keypoints;
}

cv::Mat ORBDescriptor(cv::Mat img, std::vector<cv::KeyPoint> keypoints)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(4000);
    cv::Mat descriptors;
    orb->compute(img, keypoints, descriptors);

    return descriptors;

}

std::vector<cv::DMatch> ORBDescriptorMatch(cv::Mat queryDescriptor, cv::Mat trainDescriptor)
{
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;               
    matcher->match(queryDescriptor, trainDescriptor, matches);

    return matches;

}

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

std::vector<cv::Point3f> Sort3dpointByMatch(std::vector<cv::Point3f> Disorder3dpoint, std::vector<cv::DMatch> matches)
{
    std::vector<cv::Point3f> Ordered3dpoint;
    Ordered3dpoint.resize(matches.size());
    for(size_t i = 0; i < matches.size(); i++){
        Ordered3dpoint[i] = Disorder3dpoint[matches[i].trainIdx];
    }

    return Ordered3dpoint;
}
// void CaculatePoseByPnP()
// {

// }