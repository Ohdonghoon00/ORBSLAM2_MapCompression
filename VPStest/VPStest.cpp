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

std::vector<cv::KeyPoint> VPStest::ORBFeatureExtract(cv::Mat img)
{
    std::vector<cv::KeyPoint> keypoints;
    orb->detect(img, keypoints);
    
    return keypoints;
}

cv::Mat VPStest::ORBDescriptor(cv::Mat img, std::vector<cv::KeyPoint> keypoints)
{
    cv::Mat descriptors;
    orb->compute(img, keypoints, descriptors);

    return descriptors;

}

std::vector<cv::DMatch> VPStest::ORBDescriptorMatch(cv::Mat queryDescriptor, cv::Mat trainDescriptor)
{
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;               
    matcher->match(queryDescriptor, trainDescriptor, matches);

    return matches;

}

void VPStest::SetCandidateKFid(DBoW2::QueryResults ret)
{
    CandidateKFid.clear();
    for(int i = 0; i < ret.size(); i++){
        CandidateKFid.push_back(ret[i].Id);
    }
}


// void VPStest::SetCandidateKF3dpointDB(DataBase DB)
// {
//     for(int i = 0; i < CandidateKFid.size(); i++){
//         int KFid = CandidateKFid[i];
//         std::vector<cv::Point3f> Match3dpoint = DB.GetKF3dPoint(KFid);
//         MatchDB3dPoints[KFid] = Match3dpoint;

//     }
// }



double VPStest::PnPInlierRatio(int KFid)
{
    cv::Mat R, T, RT, inliers;
    cv::solvePnPRansac(MatchDB3dPoints[KFid], MatchQ2dPoints[KFid], K, cv::noArray(), R, T, true, 100, 3.0F, 0.99, inliers, 0 );
    double PnPInlierRatio = (double)inliers.rows / (double)MatchDB3dPoints[KFid].size();

    return PnPInlierRatio;
}

// void VPStest::DescriptorMatch(DataBase DB, cv::Mat QDescriptor)
// {
//     for(int i = 0; i < CandidateKFid.size(); i ++){
//         int KFid = CandidateKFid[i];
//         cv::Mat Descriptor = DB.GetKFMatDescriptor(KFid);
//         // std::vector
//     }
// }

int VPStest::FindReferenceKF(DataBase DB, cv::Mat QDescriptor, std::vector<cv::KeyPoint> QKeypoints)
{
    std::map<int, std::vector<cv::DMatch>> MatchResults;
    std::vector<double> PnPinlierRatios;
    
    MatchDB3dPoints.clear();
    MatchQ2dPoints.clear();
    
    for(int i = 0; i < CandidateKFid.size(); i ++){
        
        int KFid = CandidateKFid[i];
        cv::Mat Descriptor = DB.GetKFMatDescriptor(KFid);
        MatchResults[KFid].resize(CandidateKFid.size());
        MatchResults[KFid] = ORBDescriptorMatch(QDescriptor, Descriptor);
        std::vector<cv::Point3f> DisOrderMatch3dpoint = DB.GetKF3dPoint(KFid);
        
        std::sort(MatchResults[KFid].begin(), MatchResults[KFid].end());
        std::vector<cv::DMatch> Goodmatches(MatchResults[KFid].begin(), MatchResults[KFid].begin() + DisOrderMatch3dpoint.size()); 

        for(int j = 0; j < Goodmatches.size(); j++){
            cv::Point3f Match3dPoint(   DisOrderMatch3dpoint[Goodmatches[j].trainIdx].x,
                                        DisOrderMatch3dpoint[Goodmatches[j].trainIdx].y,
                                        DisOrderMatch3dpoint[Goodmatches[j].trainIdx].z );

            MatchDB3dPoints[KFid].push_back(Match3dPoint);
            
            cv::Point2f MatchQ2dPoint(QKeypoints[Goodmatches[j].queryIdx].pt);
            MatchQ2dPoints[KFid].push_back(MatchQ2dPoint);
        }
        double _PnPinlierRatio = PnPInlierRatio(KFid);
        PnPinlierRatios.push_back(_PnPinlierRatio);

    }

    std::vector<double> PnPinlierRatios_(PnPinlierRatios);
    std::sort(PnPinlierRatios.begin(), PnPinlierRatios.end());
std::cout << std::endl;
    auto it = find(PnPinlierRatios_.begin(), PnPinlierRatios_.end(), PnPinlierRatios.back());
    int index = it - PnPinlierRatios_.begin();

    int ReferenceId = CandidateKFid[index];

    return ReferenceId;

}

double VPStest::VPStestToReferenceKF(DataBase DB, cv::Mat QDescriptor, std::vector<cv::KeyPoint> QKeypoints, int KFid, Eigen::Matrix4f &Pose)
{
    std::vector<cv::DMatch> Matches;
    std::vector<cv::Point3f> Match3dpts;
    std::vector<cv::Point2f> Match2dpts; 
    
    cv::Mat Descriptor = DB.GetKFMatDescriptor(KFid);
    Matches = ORBDescriptorMatch(QDescriptor, Descriptor);
    std::vector<cv::Point3f> DisOrderMatch3dpoint = DB.GetKF3dPoint(KFid);
    std::sort(Matches.begin(), Matches.end());
    std::vector<cv::DMatch> GoodMatches(Matches.begin(), Matches.begin() + DisOrderMatch3dpoint.size());

    for(int i = 0; i < GoodMatches.size(); i++){

        cv::Point3f Match3dPoint(   DisOrderMatch3dpoint[GoodMatches[i].trainIdx].x,
                                    DisOrderMatch3dpoint[GoodMatches[i].trainIdx].y,
                                    DisOrderMatch3dpoint[GoodMatches[i].trainIdx].z );

        Match3dpts.push_back(Match3dPoint);
            
        cv::Point2f MatchQ2dPoint(QKeypoints[GoodMatches[i].queryIdx].pt);
        Match2dpts.push_back(MatchQ2dPoint);

    }

    cv::Mat R, T, RT, inliers;
    cv::solvePnPRansac(Match3dpts, Match2dpts, K, cv::noArray(), R, T, true, 100, 3.0F, 0.99, inliers, 0 );
    double PnPInlierRatio = (double)inliers.rows / (double)Match3dpts.size();
    cv::Rodrigues(R, R);
    std::cout << R << std::endl;
    std::cout << T << std::endl;
    Pose << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), T.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), T.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), T.at<double>(2, 0),
            0, 0, 0, 1;
    // std::cout << Pose.inverse() << std::endl;
    Pose = Pose.inverse();
    return PnPInlierRatio;

}