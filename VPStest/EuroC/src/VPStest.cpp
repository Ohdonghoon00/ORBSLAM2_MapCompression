#include "VPStest.h"

using namespace cv;
cv::Mat K = GetK(IntrinsicData);

int VPStest::LoadDBfile(std::string dbFilePath, DataBase *DB)
{
    std::ifstream in(dbFilePath, std::ios_base::binary);
    if (!in){
        std::cout << "Cannot DataBase bin file is empty!" << std::endl;
        return EXIT_FAILURE;
    }
    boost::archive::binary_iarchive ia(in, boost::archive::no_header);
    ia >> DB;
    in.close();    
}

void VPStest::InputDBdescriptorTovoc(DataBase *DB, OrbDatabase *db)
{
    for(size_t i = 0; i < DB->KFtoMPIdx.size(); i++){

        std::cout << " KF num : " << i << "     Keypoint num : " << (DB->GetKFMatDescriptor(i)).size() << std::endl;  
        
        // Input Voc
        // cv::Mat DB_image = DB->LeftKFimg[i];
        // cv::Mat mask_, DBDescriptor_;
        // std::vector<cv::KeyPoint> DBKeypoints_;
        // cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures);
        // orb->detectAndCompute(DB_image, mask_, DBKeypoints_, DBDescriptor_);
        // ORBfeatureAndDescriptor(DB_image, mask_, DBKeypoints_, DBDescriptor_);        
        std::vector<cv::Mat> DBDescriptors = Converter::MatToVectorMat(DB->dbow2Descriptors[i]);
        db->add(DBDescriptors);
    }    
}

int VPStest::Loadgt(std::string queryGtTrajectoryPath)
{
    std::ifstream queryGtTrajectoryFile(queryGtTrajectoryPath, std::ifstream::in);

    if(!queryGtTrajectoryFile.is_open()){
        std::cout << " GT Query Trajectory file failed to open " << std::endl;
        return EXIT_FAILURE;
    }      

    std::string line;
    while(std::getline(queryGtTrajectoryFile, line)){
        
        std::string value;
        std::vector<std::string> values;

        std::stringstream ss(line);
        while(std::getline(ss, value, ' '))
            values.push_back(value);

        Vector6d pose;
        pose << std::stod(values[1]), std::stod(values[2]), std::stod(values[3]),
                std::stod(values[4]), std::stod(values[5]), std::stod(values[6]);

        gtPoses.push_back(pose);        
    }
}

void VPStest::InputQuerydb(QueryDB *query, std::string timeStampfilePath, std::string queryImgdirPath)
{
    std::ifstream s;
    s.open(timeStampfilePath);
    std::string line;
        
    // Load Query Img
    // std::cout << " Input Query Img " << std::endl;

    while(std::getline(s, line)){
        
        std::stringstream ss;
        ss << line;
        std::string QueryImgsPath = queryImgdirPath + "/" + ss.str() +".png";
        cv::Mat img = cv::imread(QueryImgsPath);
        query->qImgs.push_back(img);
        query->qTimestamps.push_back(std::stod(line) * 10e-10);
    }
    
    s.close();     
}


cv::Mat VPStest::InputQueryImg(std::string QueryFile)
{ 
    cv::Mat image;
    cv::VideoCapture video;
    if(!video.open(QueryFile)){
        std::cout << " No query image " << std::endl;
        
    }
    video >> image;

    return image;
}

void VPStest::InputQueryImg(QueryDB *query, int imageNum)
{
    query->qImg = query->qImgs[imageNum].clone();
    if(query->qImg.empty()) {
        std::cout << " Error at input query img " << std::endl; 
        // break;
    }
    if (query->qImg.channels() > 1) cv::cvtColor(query->qImg, query->qImg, cv::COLOR_RGB2GRAY);
    qImgNum = imageNum;
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
    Eigen::Matrix4Xf WorldPoints = Converter::HomogeneousForm(WPts);
    Eigen::Matrix3Xf ImagePoints = Converter::HomogeneousForm(ImgPts);

    Eigen::Matrix3Xf ReprojectPoints(3, WorldPoints.cols());
    Pose = Pose.inverse();
    Eigen::Matrix4f Pose_ = Pose.cast<float>();
    Eigen::Matrix<float, 3, 4> PoseRT;
    PoseRT = Pose_.block<3, 4>(0, 0);
    Eigen::MatrixXf K_ = Converter::Mat2Eigen(K);
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

int VPStest::FindReferenceKF(DataBase* DB, QueryDB query)
{
    std::map<int, std::vector<cv::DMatch>> MatchResults;
    std::vector<double> PnPinlierRatios;
    std::vector<int> inlier_nums;
    std::vector<cv::Mat> inliersVec;
    MatchResults.clear();
    MatchDB3dPoints.clear();
    MatchQ2dPoints.clear();
    candidatesPoses.clear();
    for(int i = 0; i < CandidateKFid.size(); i ++){
        
        int KFid = CandidateKFid[i];
        
        // test optical flow
        std::vector<cv::Point2f> DB2dpts = DB->GetKF2fpts(KFid);
        std::vector<cv::Point2f> q2fpts;
        TrackOpticalFlow(DB->LeftKFimg[KFid], query.qImg, DB2dpts, q2fpts);

        cv::Mat Descriptor = DB->GetKFMatDescriptor(KFid);
        // MatchResults[KFid].resize(CandidateKFid.size());
        MatchResults[KFid] = ORBDescriptorMatch(query.qDescriptor, Descriptor);
        std::vector<cv::Point3f> DisOrderMatch3dpoint = DB->GetKF3dPoint(KFid);
        int GoodMatchNum = DisOrderMatch3dpoint.size();
        if(DisOrderMatch3dpoint.size() > query.qDescriptor.rows) GoodMatchNum = query.qDescriptor.rows;
        std::sort(MatchResults[KFid].begin(), MatchResults[KFid].end());
        std::vector<cv::DMatch> Goodmatches(MatchResults[KFid].begin(), MatchResults[KFid].begin() + GoodMatchNum); 

        for(int j = 0; j < Goodmatches.size(); j++){
            cv::Point3f Match3dPoint(   DisOrderMatch3dpoint[Goodmatches[j].trainIdx].x,
                                        DisOrderMatch3dpoint[Goodmatches[j].trainIdx].y,
                                        DisOrderMatch3dpoint[Goodmatches[j].trainIdx].z );

            MatchDB3dPoints[KFid].push_back(Match3dPoint);
            
            cv::Point2f MatchQ2dPoint(query.qKeypoints[Goodmatches[j].queryIdx].pt);
            MatchQ2dPoints[KFid].push_back(MatchQ2dPoint);
        }
        cv::Mat R, T, RT, inliers;
        cv::solvePnPRansac(MatchDB3dPoints[KFid], MatchQ2dPoints[KFid], K, cv::noArray(), R, T, false, 1000, 3.0F, 0.99, inliers, 0 );
        double _PnPInlierRatio = (double)inliers.rows / (double)MatchDB3dPoints[KFid].size();

        Vector6d Pose = ProjectionTo6DOFPoses(R, T);
        candidatesPoses.push_back(Pose);
        
        // candidates pose err
        double err[2];
        RMSError(gtPoses[qImgNum], Pose, &err[0]);
        double err_[2];
        RMSError(DB->kfPoses[KFid], Pose, &err_[0]);
        std::cout << err[0] << "  " << Rad2Degree(err[1]) << " *** " << err_[0] << "  " << Rad2Degree(err_[1]) << std::endl;

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

    if(MaxRatioIdx == MaxInlierIdx){
        if(MaxInlier > 30)
            return CandidateKFid[MaxInlierIdx];
        else
            return -1;
    }
    else{
        if(inlier_nums[MaxRatioIdx] > 30 && MaxRatio > 0.1)
            return CandidateKFid[MaxRatioIdx];
        else
            return -1;
    }


    // if( MaxInlier > 30)
    //     return CandidateKFid[MaxInlierIdx];
    // else{
    //     // std::vector<cv::KeyPoint> DB2dMatchForDraw = DB->GetKFkeypoint(CandidateKFid[MaxInlierIdx]);
    //     // cv::Mat matchimg;
    //     // std::vector<cv::DMatch> Goodmatches_(MatchResults[CandidateKFid[MaxInlierIdx]].begin(), MatchResults[CandidateKFid[MaxInlierIdx]].begin() + 30); 
    //     // cv::drawMatches(Qimg, QKeypoints, DB->LeftKFimg[CandidateKFid[MaxInlierIdx]], DB2dMatchForDraw, Goodmatches_, matchimg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DEFAULT);
    //     // cv::imshow("matchtop_", matchimg);
        
    //     // std::vector<cv::DMatch> Inliermatches = MatchResults[CandidateKFid[MaxInlierIdx]];
    //     // InlierMatchResult(Inliermatches, inliersVec[MaxInlierIdx]);
    //     // cv::Mat InlierMatchImg;
    //     // cv::drawMatches(Qimg, QKeypoints, DB->LeftKFimg[CandidateKFid[MaxInlierIdx]], DB2dMatchForDraw, Inliermatches, InlierMatchImg, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DEFAULT);
    //     // cv::imshow("inlierMatch", InlierMatchImg);
    //     return -1;
    // }
    // // Place Recognition debug
    // cv::Mat BestRatioImage = DB->LeftKFimg[CandidateKFid[MaxRatioIdx]];
    // cv::imshow("BestRatioImage", BestRatioImage);
    
    // return ReferenceId;

}

double VPStest::VPStestToReferenceKF(DataBase* DB, QueryDB query, int KFid, Eigen::Matrix4d &Pose, cv::Mat &Inliers, std::vector<cv::DMatch> &GoodMatches_)
{
    std::vector<cv::DMatch> Matches;
    std::vector<cv::Point3f> Match3dpts;
    std::vector<cv::Point2f> Match2dpts;
    // std::vector<cv::KeyPoint> DB2dpts;
    
    int NearSearchNum = 0;
    float Disthres = 100.0;

    cv::Mat Descriptor = DB->GetKFMatDescriptor(KFid);
    // cv::Mat Descriptor = DB->GetNearReferenceKFMatDescriptor(KFid, NearSearchNum);
    
    Matches = ORBDescriptorMatch(query.qDescriptor, Descriptor);
    
    totalLandmarks.clear();
    totalLandmarks = DB->GetKF3dPoint(KFid);
    // std::vector<cv::Point3f> totalLandmarks = DB->GetNearReferenceKF3dPoint(KFid, NearSearchNum);
    int GoodMatchNum = totalLandmarks.size();
    if(totalLandmarks.size() > query.qDescriptor.rows) GoodMatchNum = query.qDescriptor.rows;
    
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
    
    qTotal2fpts.clear();
    qTotal2fpts = Converter::KeyPoint2Point2f(query.qKeypoints);
    // DB2dpts = DB->GetKFkeypoint(KFid);
    std::cout << " After erase match size : " << GoodMatches.size() << std::endl;
    for(int i = 0; i < GoodMatches.size(); i++){
        // std::cout << GoodMatches[i].distance << " ";
        cv::Point3f Match3dPoint(   totalLandmarks[GoodMatches[i].trainIdx].x,
                                    totalLandmarks[GoodMatches[i].trainIdx].y,
                                    totalLandmarks[GoodMatches[i].trainIdx].z );

        Match3dpts.push_back(Match3dPoint);
            
        cv::Point2f MatchQ2dPoint(  qTotal2fpts[GoodMatches[i].queryIdx].x,
                                    qTotal2fpts[GoodMatches[i].queryIdx].y);
        Match2dpts.push_back(MatchQ2dPoint);

    }
    GoodMatches_ = GoodMatches;


    cv::Mat R, T, RT;
    double PnPInlierRatio = 0;
    bool PnPSuccess = cv::solvePnPRansac(Match3dpts, Match2dpts, K, cv::noArray(), R, T, false, 1000, 3.0F, 0.99, Inliers, 0 );
    // inlier_num = Inliers.rows;
    std::cout << " SolvePnP result  : " << PnPSuccess << std::endl;
    
        
        PnPInlierRatio = (double)Inliers.rows / (double)Match3dpts.size();
        
        queryPose = ProjectionTo6DOFPoses(R, T);
        queryPose4d = To44RT(queryPose);
        Pose = queryPose4d;

    
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

    inlierLandmarks.clear();
    qInlier2fpts.clear();
    std::vector<cv::DMatch> Matches_;

    for(int i = 0; i < Inliers.rows; i++){
        int index = Inliers.at<int>(i, 0);
        cv::Point3f Match3dPoint(   totalLandmarks[Matches[index].trainIdx].x,
                                    totalLandmarks[Matches[index].trainIdx].y,
                                    totalLandmarks[Matches[index].trainIdx].z );                                    

        inlierLandmarks.push_back(Match3dPoint);
        
        cv::Point2f MatchQ2dPoint(  qTotal2fpts[Matches[index].queryIdx].x,
                                    qTotal2fpts[Matches[index].queryIdx].y);
        qInlier2fpts.push_back(MatchQ2dPoint);
        
        Matches_.push_back(Matches[index]);
    }
    
    ReprojectionError(inlierLandmarks, qInlier2fpts, queryPose4d);

    Matches.clear();
    Matches = Matches_;
}

std::vector<float> VPStest::ReprojectionError(std::vector<cv::Point3f> WPts, std::vector<cv::Point2f> ImgPts, Eigen::Matrix4d Pose)
{
    projection2fpts.clear();

    Eigen::Matrix4Xf WorldPoints = Converter::HomogeneousForm(WPts);
    Eigen::Matrix3Xf ImagePoints = Converter::HomogeneousForm(ImgPts);

    Eigen::Matrix3Xf ReprojectPoints(3, WorldPoints.cols());
    Pose = Pose.inverse();
    Eigen::Matrix4f Pose_ = Pose.cast<float>();
    Eigen::Matrix<float, 3, 4> PoseRT;
    PoseRT = Pose_.block<3, 4>(0, 0);
    Eigen::MatrixXf K_ = Converter::Mat2Eigen(K);
    ReprojectPoints = PoseRT * WorldPoints;

    for(int i = 0; i < ReprojectPoints.cols(); i++){
        ReprojectPoints(0, i) /= ReprojectPoints(2, i);
        ReprojectPoints(1, i) /= ReprojectPoints(2, i);
        ReprojectPoints(2, i) /= ReprojectPoints(2, i);
    }
    
    ReprojectPoints = K_ * ReprojectPoints;
    
    std::vector<float> ReprojectErr(WorldPoints.cols());
    for(int i = 0; i < WorldPoints.cols(); i++){
        ReprojectErr[i] = std::sqrt( (ImagePoints(0, i) - ReprojectPoints(0, i)) * 
                                     (ImagePoints(0, i) - ReprojectPoints(0, i)) + 
                                     (ImagePoints(1, i) - ReprojectPoints(1, i)) *
                                     (ImagePoints(1, i) - ReprojectPoints(1, i)) );
        // std::cout << ReprojectErr[i] << " ";
        cv::Point2f prof2fpts(ReprojectPoints(0, i), ReprojectPoints(1, i));
        projection2fpts.push_back(prof2fpts);
    }
    std::cout << std::endl;

    return ReprojectErr;
}

void VPStest::RMSError(Vector6d EsPose, Vector6d gtPose, double *err)
{
    // err [0] -> trans , [1] -> rot
   
    // RSE Error (root - square error)
    Eigen::Matrix4d RelativePose = To44RT(gtPose).inverse() * To44RT(EsPose);
        
    // trans
    Eigen::Vector3d RelativeTrans;
    RelativeTrans << RelativePose(0, 3), RelativePose(1, 3), RelativePose(2, 3);
    err[0] = std::sqrt(RelativeTrans.dot(RelativeTrans));
        
    // rotation
    Eigen::Matrix3d RelativeRot_ = RelativePose.block<3, 3>(0, 0);
    Eigen::Vector3d RelativeRot = ToVec3(RelativeRot_);
    err[1] = std::sqrt(RelativeRot.dot(RelativeRot));

}

void VPStest::TrackOpticalFlow(cv::Mat previous, cv::Mat current, std::vector<cv::Point2f> &previous_pts, std::vector<cv::Point2f> &current_pts)
{
    std::vector<uchar> status;
    cv::Mat err;

    cv::calcOpticalFlowPyrLK(previous, current, previous_pts, current_pts, status, err);


    const int image_x_size_ = previous.cols;
    const int image_y_size_ = previous.rows;

    // remove err point
    int indexCorrection = 0;

    for( int i = 0; i < status.size(); i++)
    {
        cv::Point2f pt = current_pts.at(i- indexCorrection);
        if((pt.x < 0)||(pt.y < 0 )||(pt.x > image_x_size_)||(pt.y > image_y_size_)) status[i] = 0;
        if (status[i] == 0)	
        {
                    
                    previous_pts.erase ( previous_pts.begin() + i - indexCorrection);
                    current_pts.erase (current_pts.begin() + i - indexCorrection);
                    indexCorrection++;
        }

    }   
}