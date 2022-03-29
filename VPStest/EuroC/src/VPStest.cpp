#include "VPStest.h"

using namespace cv;
cv::Mat K = GetK(IntrinsicData);

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
    for(int i = 0; i < 20; i++){
        CandidateKFid.push_back(ret[i].Id);
    }
}



double VPStest::PnPInlierRatio(int KFid)
{
    cv::Mat R, T, RT, inliers;
    
    // std::cout << MatchDB3dPoints[KFid].size() << "  " << MatchQ2dPoints[KFid].size() << std::endl;
    cv::solvePnPRansac(MatchDB3dPoints[KFid], MatchQ2dPoints[KFid], K, cv::noArray(), R, T, false, 300, 3.0F, 0.99, inliers, 0 );
    double PnPInlierRatio = (double)inliers.rows / (double)MatchDB3dPoints[KFid].size();

    return PnPInlierRatio;
}

std::vector<float> ReprojectionError(std::vector<cv::Point3f> WPts, std::vector<cv::Point2f> ImgPts, Eigen::Matrix4d Pose)
{
    Eigen::Matrix4Xf WorldPoints = HomogeneousForm(WPts);
    Eigen::Matrix3Xf ImagePoints = HomogeneousForm(ImgPts);

    Eigen::Matrix3Xf ReprojectPoints(3, WorldPoints.cols());
    Pose = Pose.inverse();
    Eigen::Matrix4f Pose_ = Pose.cast<float>();
    Eigen::Matrix<float, 3, 4> PoseRT;
    PoseRT = Pose_.block<3, 4>(0, 0);
    Eigen::MatrixXf K_ = Mat2Eigen(K);
    ReprojectPoints = PoseRT * WorldPoints;
    for(int i = 0; i < ReprojectPoints.cols(); i++){
        ReprojectPoints(0, i) /= ReprojectPoints(2, i);
        ReprojectPoints(1, i) /= ReprojectPoints(2, i);
        ReprojectPoints(2, i) /= ReprojectPoints(2, i);
    }
    ReprojectPoints = K_ * ReprojectPoints;
    // for(int i = 0; i < ReprojectPoints.cols(); i++){
    //     std::cout << ReprojectPoints(0, i) << " " << ReprojectPoints(1, i) << " " << ReprojectPoints(2, i);
    // }

    std::vector<float> ReprojectErr(WorldPoints.cols());
    for(int i = 0; i < WorldPoints.cols(); i++){
        ReprojectErr[i] = std::sqrt( (ImagePoints(0, i) - ReprojectPoints(0, i)) * 
                                     (ImagePoints(0, i) - ReprojectPoints(0, i)) + 
                                     (ImagePoints(1, i) - ReprojectPoints(1, i)) *
                                     (ImagePoints(1, i) - ReprojectPoints(1, i)) );
        // std::cout << ReprojectErr[i] << " ";
    }
    std::cout << std::endl;

    return ReprojectErr;
}

int VPStest::FindReferenceKF(DataBase* DB, cv::Mat QDescriptor, std::vector<cv::KeyPoint> QKeypoints, cv::Mat Qimg)
{
    std::map<int, std::vector<cv::DMatch>> MatchResults;
    std::vector<double> PnPinlierRatios;
    std::vector<int> inlier_nums;
    std::vector<cv::Mat> inliersVec;
    MatchResults.clear();
    MatchDB3dPoints.clear();
    MatchQ2dPoints.clear();
    for(int i = 0; i < CandidateKFid.size(); i ++){
        
        int KFid = CandidateKFid[i];
        cv::Mat Descriptor = DB->GetKFMatDescriptor(KFid);
        // MatchResults[KFid].resize(CandidateKFid.size());
        MatchResults[KFid] = ORBDescriptorMatch(QDescriptor, Descriptor);
        std::vector<cv::Point3f> DisOrderMatch3dpoint = DB->GetKF3dPoint(KFid);
        int GoodMatchNum = DisOrderMatch3dpoint.size();
        if(DisOrderMatch3dpoint.size() > QDescriptor.rows) GoodMatchNum = QDescriptor.rows;
        std::sort(MatchResults[KFid].begin(), MatchResults[KFid].end());
        std::vector<cv::DMatch> Goodmatches(MatchResults[KFid].begin(), MatchResults[KFid].begin() + GoodMatchNum); 

        for(int j = 0; j < Goodmatches.size(); j++){
            cv::Point3f Match3dPoint(   DisOrderMatch3dpoint[Goodmatches[j].trainIdx].x,
                                        DisOrderMatch3dpoint[Goodmatches[j].trainIdx].y,
                                        DisOrderMatch3dpoint[Goodmatches[j].trainIdx].z );

            MatchDB3dPoints[KFid].push_back(Match3dPoint);
            
            cv::Point2f MatchQ2dPoint(QKeypoints[Goodmatches[j].queryIdx].pt);
            MatchQ2dPoints[KFid].push_back(MatchQ2dPoint);
        }
        cv::Mat R, T, RT, inliers;
        cv::solvePnPRansac(MatchDB3dPoints[KFid], MatchQ2dPoints[KFid], K, cv::noArray(), R, T, false, 1000, 3.0F, 0.99, inliers, 0 );
        double _PnPInlierRatio = (double)inliers.rows / (double)MatchDB3dPoints[KFid].size();
        // double _PnPinlierRatio = PnPInlierRatio(KFid);
        PnPinlierRatios.push_back(_PnPInlierRatio);
        inlier_nums.push_back(inliers.rows);
        inliersVec.push_back(inliers);



    }

    // print value
    for(int i = 0; i < PnPinlierRatios.size(); i++) std::cout << PnPinlierRatios[i] << "  ";
    std::cout << std::endl;
    for(int i = 0; i < inlier_nums.size(); i++) std::cout << inlier_nums[i] << "  ";
    std::cout << std::endl;

    double MaxRatio = *max_element(PnPinlierRatios.begin(), PnPinlierRatios.end());
    int MaxInlier = *max_element(inlier_nums.begin(), inlier_nums.end());
    
    int MaxRatioIdx = max_element(PnPinlierRatios.begin(), PnPinlierRatios.end()) - PnPinlierRatios.begin();
    int MaxInlierIdx = max_element(inlier_nums.begin(), inlier_nums.end()) - inlier_nums.begin();
    std::cout << MaxInlier << std::endl;
    std::cout << MaxInlierIdx << std::endl;


    if( MaxInlier > 30)
        return CandidateKFid[MaxInlierIdx];
    else{
        // std::vector<cv::KeyPoint> DB2dMatchForDraw = DB->GetKF2dPoint(CandidateKFid[MaxInlierIdx]);
        // cv::Mat matchimg;
        // std::vector<cv::DMatch> Goodmatches_(MatchResults[CandidateKFid[MaxInlierIdx]].begin(), MatchResults[CandidateKFid[MaxInlierIdx]].begin() + 30); 
        // cv::drawMatches(Qimg, QKeypoints, DB->LeftKFimg[CandidateKFid[MaxInlierIdx]], DB2dMatchForDraw, Goodmatches_, matchimg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DEFAULT);
        // cv::imshow("matchtop_", matchimg);
        
        // std::vector<cv::DMatch> Inliermatches = MatchResults[CandidateKFid[MaxInlierIdx]];
        // InlierMatchResult(Inliermatches, inliersVec[MaxInlierIdx]);
        // cv::Mat InlierMatchImg;
        // cv::drawMatches(Qimg, QKeypoints, DB->LeftKFimg[CandidateKFid[MaxInlierIdx]], DB2dMatchForDraw, Inliermatches, InlierMatchImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DEFAULT);
        // cv::imshow("inlierMatch", InlierMatchImg);
        return -1;
    }
    // // Place Recognition debug
    // cv::Mat BestRatioImage = DB->LeftKFimg[CandidateKFid[MaxRatioIdx]];
    // cv::imshow("BestRatioImage", BestRatioImage);
    
    // return ReferenceId;

}

double VPStest::VPStestToReferenceKF(DataBase* DB, cv::Mat QDescriptor, std::vector<cv::KeyPoint> QKeypoints, int KFid, Eigen::Matrix4d &Pose, cv::Mat &Inliers, std::vector<cv::DMatch> &GoodMatches_)
{
    std::vector<cv::DMatch> Matches;
    std::vector<cv::Point3f> Match3dpts;
    std::vector<cv::Point2f> Match2dpts;
    std::vector<cv::KeyPoint> DB2dpts;
    
    int NearSearchNum = 0;
    float Disthres = 40.0;

    cv::Mat Descriptor = DB->GetKFMatDescriptor(KFid);
    // cv::Mat Descriptor = DB->GetNearReferenceKFMatDescriptor(KFid, NearSearchNum);
    
    Matches = ORBDescriptorMatch(QDescriptor, Descriptor);
    
    std::vector<cv::Point3f> DisOrderMatch3dpoint = DB->GetKF3dPoint(KFid);
    // std::vector<cv::Point3f> DisOrderMatch3dpoint = DB->GetNearReferenceKF3dPoint(KFid, NearSearchNum);
    int GoodMatchNum = DisOrderMatch3dpoint.size();
    if(DisOrderMatch3dpoint.size() > QDescriptor.rows) GoodMatchNum = QDescriptor.rows;
    
    std::sort(Matches.begin(), Matches.end());
    std::vector<cv::DMatch> GoodMatches(Matches.begin(), Matches.begin() + GoodMatchNum);
    
    std::cout << " match size : " << GoodMatches.size() << std::endl;
    int k = GoodMatches.size();
    int index = 0;
    for(int i = 0; i < k; i++){
        float distance = GoodMatches[i].distance;
        if(distance > Disthres){
            GoodMatches.erase(GoodMatches.begin() + i - index);
            index++;
        }
    }
    
    DB2dpts = DB->GetKF2dPoint(KFid);
    std::cout << " After erase match size : " << GoodMatches.size() << std::endl;
    for(int i = 0; i < GoodMatches.size(); i++){
        // std::cout << GoodMatches[i].distance << " ";
        cv::Point3f Match3dPoint(   DisOrderMatch3dpoint[GoodMatches[i].trainIdx].x,
                                    DisOrderMatch3dpoint[GoodMatches[i].trainIdx].y,
                                    DisOrderMatch3dpoint[GoodMatches[i].trainIdx].z );

        Match3dpts.push_back(Match3dPoint);
            
        cv::Point2f MatchQ2dPoint(QKeypoints[GoodMatches[i].queryIdx].pt);
        Match2dpts.push_back(MatchQ2dPoint);

    }

    GoodMatches_ = GoodMatches;

    cv::Mat R, T, RT;
    double PnPInlierRatio = 0;
    bool PnPSuccess = cv::solvePnPRansac(Match3dpts, Match2dpts, K, cv::noArray(), R, T, false, 1000, 3.0F, 0.99, Inliers, 0 );
    // inlier_num = Inliers.rows;
    std::cout << " SolvePnP result  : " << PnPSuccess << std::endl;
    
        
        PnPInlierRatio = (double)Inliers.rows / (double)Match3dpts.size();
        cv::Rodrigues(R, R);
        Pose << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), T.at<double>(0, 0),
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), T.at<double>(1, 0),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), T.at<double>(2, 0),
                0, 0, 0, 1;
        Pose = Pose.inverse();
    
        // std::vector<float> ReprojectionErrViz = ReprojectionError(Match3dpts, Match2dpts, Pose);
        // for(int i = 0; i < Inliers.rows; i++){
        //     int index = Inliers.at<int>(i, 0);
        //     std::cout << ReprojectionErrViz[index] << " ";
        // }
    
    return PnPInlierRatio;

}

int VPStest::FindKFImageNum(int KFid, DataBase* DB, std::vector<double> timestamps)
{
    double DBtimestamp = DB->timestamps[KFid];
    auto it = find(timestamps.begin(), timestamps.end(), DBtimestamp);
    int index = 0;
    if(it != timestamps.end()){
        index = it - timestamps.begin();
    }

    return index;
}

void VPStest::InlierMatchResult(std::vector<cv::DMatch> &Matches, cv::Mat Inliers)
{

    
    std::vector<cv::KeyPoint> QMatch2dpts;
    std::vector<cv::DMatch> Matches_;

    for(int i = 0; i < Inliers.rows; i++){
        int index = Inliers.at<int>(i, 0);
                                    

            
        Matches_.push_back(Matches[index]);
    }
    
    Matches.clear();
    Matches = Matches_;
}