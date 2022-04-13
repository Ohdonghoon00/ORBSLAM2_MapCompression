#include "utils.h"

//////////////////////////////////////////////////////////
Eigen::Vector3d Origin{0.0, 0.0, 0.0};
Eigen::Vector3d ZVec = Eigen::Vector3d::UnitZ();
Eigen::Matrix3d Iden = Eigen::Matrix3d::Identity();

//////////// EuroC ////////////////////

double fx(435.2046959714599), fy(435.2046959714599), cx(367.4517211914062), cy(252.2008514404297);
double IntrinsicData[] = {   fx, 0.0,cx, 
                            0.0, fy, cy,
                            0.0, 0.0, 1.0}; 
float fx_(435.2046959714599), fy_(435.2046959714599), cx_(367.4517211914062), cy_(252.2008514404297);
float IntrinsicData_[] = {   fx_, 0.0,cx_, 
                            0.0, fy_, cy_,
                            0.0, 0.0, 1.0};

///// Extrinsic /////
// Machine Hall  body(IMU) - cam0 //
double Cam0ToBodyData[] = {0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0};

// Machine Hall  body(IMU) - cam1 //
double Cam1ToBodyData[] = {0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
         0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
        -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
         0.0, 0.0, 0.0, 1.0};


cv::Mat K = GetK(IntrinsicData);
cv::Mat Kf = GetKf(IntrinsicData_);
cv::Point2d c(cx, cy); 


// ViconRoom1  body(IMU) - cam0 //



// ViconRoom2  body(IMU) - cam0 //






//////////////////////////////////////////////////////////

cv::Mat GetK(double* IntrinsicData)
{
    cv::Mat K(3, 3, CV_64FC1, IntrinsicData);
    return K;
}

cv::Mat GetKf(float* IntrinsicData)
{
    cv::Mat K(3, 3, CV_32FC1, IntrinsicData);
    return K;
}

Eigen::Matrix4d GetCam2Body(double * Cam2BodyData)
{
    Eigen::Matrix4d Cam2Body = Eigen::Map<Eigen::Matrix4d>(Cam2BodyData);
    return Cam2Body.transpose();
}

Eigen::Matrix4d GetCam1ToCam0(double * Cam2BodyData0, double * Cam2BodyData1)
{
    Eigen::Matrix4d Cam0ToBody = Eigen::Map<Eigen::Matrix4d>(Cam2BodyData0);
    Eigen::Matrix4d Cam0ToBody_ = Cam0ToBody.transpose();
    
    Eigen::Matrix4d Cam1ToBody = Eigen::Map<Eigen::Matrix4d>(Cam2BodyData1);
    Eigen::Matrix4d Cam1ToBody_ = Cam1ToBody.transpose();
    
    Eigen::Matrix4d Cam1ToCam0 = Cam0ToBody_.inverse() * Cam1ToBody_;
    return Cam1ToCam0;
}

int ReadgtPose(const std::string gtpath, std::vector<Vector6d>* poses)
{
    std::ifstream gtFile(gtpath, std::ifstream::in);
    if(!gtFile.is_open()){
        std::cout << " gtpose file failed to open " << std::endl;
        return EXIT_FAILURE;
    }

    std::string line;
    while(std::getline(gtFile, line)){
        std::string value;
        std::vector<std::string> values;

        std::stringstream ss(line);
        while(std::getline(ss, value, ' '))
            values.push_back(value);
        
        Vector6d pose;
        pose << std::stod(values[1]), std::stod(values[2]), std::stod(values[3]), std::stod(values[4]), std::stod(values[5]), std::stod(values[6]);
        poses->push_back(pose);
    }       

}


std::vector<Eigen::Vector3d> Mat3XdToVec3d(Eigen::Matrix3Xd LidarPoints)
{
    std::vector<Eigen::Vector3d> PointCloud(LidarPoints.cols());
    for(int i = 0; i < LidarPoints.cols(); i++){
        PointCloud[i].x() = LidarPoints(0, i);
        PointCloud[i].y() = LidarPoints(1, i);
        PointCloud[i].z() = LidarPoints(2, i);
    }

    return PointCloud;
}



Eigen::Vector3d ToVec3(Eigen::Matrix3d rot)
{
    Eigen::AngleAxisd rod(rot);
    Eigen::Vector3d axis(rod.axis());
    double angle = rod.angle();
    axis *= angle;

    Eigen::Vector3d vec3;
    vec3 << axis.x(), axis.y(), axis.z();

    return vec3;
}

Eigen::Vector3f ToVec3(Eigen::Matrix3f rot)
{
    Eigen::AngleAxisf rod(rot);
    Eigen::Vector3f axis(rod.axis());
    float angle = rod.angle();
    axis *= angle;

    Eigen::Vector3f vec3;
    vec3 << axis.x(), axis.y(), axis.z();

    return vec3;
}

Eigen::Matrix3d ToMat33(Eigen::Vector3d rod)
{
    Eigen::AngleAxisd r(rod.norm(), rod.normalized());
    Eigen::Matrix3d rot = r.toRotationMatrix();

    return rot;
}

Eigen::Matrix3f ToMat33(Eigen::Vector3f rod)
{
    Eigen::AngleAxisf r(rod.norm(), rod.normalized());
    Eigen::Matrix3f rot = r.toRotationMatrix();

    return rot;
}

Vector6d To6DOF(Eigen::Quaterniond q, Eigen::Vector3d t)
{
    Eigen::AngleAxisd rod(q);
    Eigen::Vector3d r(rod.axis());
    double angle = rod.angle();
    r *= angle;

    Vector6d Pose;
    Pose << r.x(), r.y(), r.z(), t.x(), t.y(), t.z();

    return Pose;
}

Vector6d To6DOF(Eigen::Matrix4d RT)
{
    Eigen::Matrix3d R = RT.block<3, 3>(0, 0);
    Eigen::Vector3d rod = ToVec3(R);
    
    Vector6d Pose;
    Pose << rod.x(), rod.y(), rod.z(), RT(0, 3), RT(1, 3), RT(2, 3);
    return Pose;
}

Eigen::Quaterniond ToQuaternion(const Vector6d Pose)
{
    Eigen::Matrix4d Pos = To44RT(Pose);
    Eigen::Matrix3d rot = Pos.block<3, 3>(0, 0); 

    Eigen::Quaterniond q(rot);

    return q; 
}



Eigen::Matrix4f To44RT(Vector6f pose)
{
    Eigen::Vector3f rod;
    rod << pose[0], pose[1], pose[2];
    Eigen::Matrix3f rot = ToMat33(rod);

    Eigen::Matrix4f RT;
    RT <<   rot(0, 0), rot(0, 1), rot(0, 2), pose[3],
            rot(1, 0), rot(1, 1), rot(1, 2), pose[4],
            rot(2, 0), rot(2, 1), rot(2, 2), pose[5],
            0,         0,         0,         1;

    return RT;
}

Eigen::Matrix4d To44RT(Vector6d pose)
{
    Eigen::Vector3d rod;
    rod << pose[0], pose[1], pose[2];
    Eigen::Matrix3d rot = ToMat33(rod);

    Eigen::Matrix4d RT;
    RT <<   rot(0, 0), rot(0, 1), rot(0, 2), pose[3],
            rot(1, 0), rot(1, 1), rot(1, 2), pose[4],
            rot(2, 0), rot(2, 1), rot(2, 2), pose[5],
            0,         0,         0,         1;

    return RT;
}

Eigen::Matrix4d To44RT(std::vector<double> pose)
{
    Eigen::Vector3d rod;
    rod << pose[0], pose[1], pose[2];

    Eigen::Matrix3d R = ToMat33(rod);

    Eigen::Matrix4d RT;
    RT <<   R(0, 0), R(0, 1), R(0, 2), pose[3],
            R(1, 0), R(1, 1), R(1, 2), pose[4],
            R(2, 0), R(2, 1), R(2, 2), pose[5],
            0,       0,       0,       1;

    return RT;
}

cv::Mat Vec6To34ProjMat(Vector6d pose)
{
    Eigen::Matrix4d CamPose = To44RT(pose);
    std::cout << "lpose : " << CamPose << std::endl;
    Eigen::Matrix4d CamProj = CamPose.inverse();
    double data[] = {   CamProj(0, 0), CamProj(0, 1), CamProj(0, 2), CamProj(0, 3),
                        CamProj(1, 0), CamProj(1, 1), CamProj(1, 2), CamProj(1, 3),
                        CamProj(2, 0), CamProj(2, 1), CamProj(2, 2), CamProj(2, 3)};
    cv::Mat Proj34(3, 4, CV_64F, data);
    return Proj34.clone();
}

cv::Mat rVec6To34ProjMat(Vector6d pose)
{
    Eigen::Matrix4d lCamPose = To44RT(pose);
    Eigen::Matrix4d rCamPose = lCamPose * GetCam1ToCam0(Cam0ToBodyData, Cam1ToBodyData);
    // rCamPose(0,3) = rCamPose(0,3) + 0.5;
    std::cout << "rpose : " << rCamPose << std::endl;
    Eigen::Matrix4d CamProj = rCamPose.inverse();
    double data[] = {   CamProj(0, 0), CamProj(0, 1), CamProj(0, 2), CamProj(0, 3),
                        CamProj(1, 0), CamProj(1, 1), CamProj(1, 2), CamProj(1, 3),
                        CamProj(2, 0), CamProj(2, 1), CamProj(2, 2), CamProj(2, 3)};    
    cv::Mat Proj34(3, 4, CV_64F, data);
    return Proj34.clone();


}

Eigen::Matrix4d Proj34ToPose(cv::Mat Proj)
{
    Eigen::Matrix4d Proj_;
    Proj_ << Proj.at<double>(0, 0), Proj.at<double>(0, 1), Proj.at<double>(0, 2), Proj.at<double>(0, 3),
            Proj.at<double>(1, 0), Proj.at<double>(1, 1), Proj.at<double>(1, 2), Proj.at<double>(1, 3),
            Proj.at<double>(2, 0), Proj.at<double>(2, 1), Proj.at<double>(2, 2), Proj.at<double>(2, 3),
            0, 0, 0, 1;
    return Proj_.inverse();
}

double ToAngle(Eigen::Matrix4d LidarRotation)
{
    Eigen::Matrix3d rot = LidarRotation.block<3, 3>(0, 0);
    Eigen::AngleAxisd rod(rot);
    double angle = rod.angle();

    return angle;
}

Eigen::Vector3d ToAxis(Eigen::Matrix4d LidarRotation)
{
    Eigen::Matrix3d rot = LidarRotation.block<3, 3>(0, 0);
    Eigen::AngleAxisd rod(rot);
    Eigen::Vector3d Axis = rod.axis();    

    return Axis;
}

Vector6d ToProjection(Vector6d pose)
{
    Eigen::Matrix4d pos = To44RT(pose);
    pos = pos.inverse();
    Vector6d proj = To6DOF(pos);
    return proj;
}

float VerticalAngle(Eigen::Vector3d p){
  return atan(p.z() / sqrt(p.x() * p.x() + p.y() * p.y())) * 180 / M_PI;
}

double PointDistance(Eigen::Vector3d p){
  return sqrt(p.x()*p.x() + p.y()*p.y() + p.z()*p.z());
}

double PointDistance(Eigen::Vector3d p1, Eigen::Vector3d p2){
  return sqrt((p1.x()-p2.x())*(p1.x()-p2.x()) + (p1.y()-p2.y())*(p1.y()-p2.y()) + (p1.z()-p2.z())*(p1.z()-p2.z()));
}

double CosRaw2(double a, double b, float ang){
    return sqrt(a * a + b * b - 2 * a * b * cos(ang * M_PI / 180));
}

double Rad2Degree(double rad){
    return rad * 180 / M_PI;
}

double Ddegree2Rad(double degree){
    return degree * M_PI / 180;
}

std::vector<cv::Point3d> ToXYZ(cv::Mat &X)
{
    std::vector<cv::Point3d> MapPts;
    X.convertTo(X, CV_64F);
    for (int i = 0 ; i < X.cols; i++)
    {
        X.col(i).row(0) = X.col(i).row(0) / X.col(i).row(3);
        X.col(i).row(1) = X.col(i).row(1) / X.col(i).row(3);
        X.col(i).row(2) = X.col(i).row(2) / X.col(i).row(3);
        X.col(i).row(3) = 1;
        MapPts.push_back(cv::Point3d(X.at<double>(0, i), X.at<double>(1, i), X.at<double>(2, i)));
    }

    return MapPts;
}
    



std::vector<float> ReprojectionError(std::vector<cv::Point3d> WPts, std::vector<cv::Point2f> ImgPts, Eigen::Matrix4d Pose)
{
    Eigen::Matrix4Xf WorldPoints = Converter::HomogeneousForm(WPts);
    Eigen::Matrix3Xf ImagePoints = Converter::HomogeneousForm(ImgPts);

    Eigen::Matrix3Xf ReprojectPoints(3, WorldPoints.cols());
    Pose = Pose.inverse();
    Eigen::Matrix4f Pose_ = Pose.cast<float>();
    Eigen::Matrix<float, 3, 4> PoseRT;
    PoseRT = Pose_.block<3, 4>(0, 0);
    Eigen::MatrixXf K_ = Converter::Mat2Eigen(Kf);
    ReprojectPoints = PoseRT * WorldPoints;

    for(int i = 0; i < ReprojectPoints.cols(); i++){
        ReprojectPoints(0, i) /= ReprojectPoints(2, i);
        ReprojectPoints(1, i) /= ReprojectPoints(2, i);
        ReprojectPoints(2, i) /= ReprojectPoints(2, i);
    }
    ReprojectPoints = K_ * ReprojectPoints;

    // for(int i = 0; i < ReprojectPoints.cols(); i++){
    //     std::cout << ReprojectPoints(0, i) << " " << ReprojectPoints(1, i) << " " << ReprojectPoints(2, i) << std::endl;
    // }
    // for(int i = 0; i < ImagePoints.cols(); i++){
    //     std::cout << ImagePoints(0, i) << " " << ImagePoints(1, i) << " " << ImagePoints(2, i) << std::endl;
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

int FindTimestampIdx(const double a, const std::vector<double> b)
{
    double MinVal = DBL_MAX;
    int MinIdx = -1;

    for(int i = 0; i < b.size(); i++){
        double diff = std::fabs(b[i] - a);
        if(diff < MinVal){
            MinVal = diff;
            MinIdx = i;
        }
    }
    return MinIdx;
}

    int readCsvGtPose(std::string gtpath, std::vector<Vector6d>* poses, std::vector<double>* timeStamps)
    {
        std::ifstream gtFile(gtpath, std::ifstream::in);
        if(!gtFile.is_open()){
            std::cout << " gtpose file failed to open " << std::endl;
            return EXIT_FAILURE;
        }

        int lineNum = 0;
        std::string line;
        while(std::getline(gtFile, line)){
            if(lineNum == 0){
                lineNum++;
                continue;
            }
            std::string value;
            std::vector<std::string> values;

            std::stringstream ss(line);
            while(std::getline(ss, value, ','))
                values.push_back(value);
            
            Eigen::Quaterniond q;
            q.x() = std::stod(values[5]);
            q.y() = std::stod(values[6]);
            q.z() = std::stod(values[7]);
            q.w() = std::stod(values[4]);

            Eigen::Vector3d t;
            t << std::stod(values[1]), std::stod(values[2]),std::stod( values[3]);
            Vector6d Pose = To6DOF(q, t);
            poses->push_back(Pose);
            // double timestamp = std::floor(std::stod(values[0]) * 1e5) * 1e-5;
            timeStamps->push_back(std::stod(values[0]));
        }       

    }


void OpticalFlowStereo(cv::Mat previous, cv::Mat current, std::vector<cv::Point2f> &previous_pts, std::vector<cv::Point2f> &current_pts)
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
    
    // int indexCorrection_ = 0;
    // int ptsNum = previous_pts.size();
    // for(int i = 0; i < ptsNum; i++){
    //     double diff_x = std::fabs(previous_pts[i - indexCorrection_].x - current_pts[i - indexCorrection_].x);
    //     double diff_y = std::fabs(previous_pts[i - indexCorrection_].y - current_pts[i - indexCorrection_].y);
    //     std::cout << diff_x << "  " << diff_y << std::endl;
    //     if(diff_x > 30 || diff_y > 15){
    //         previous_pts.erase ( previous_pts.begin() + i - indexCorrection_);
    //         current_pts.erase (current_pts.begin() + i - indexCorrection_);
    //         indexCorrection_++;           
    //     }
    // }
}

void OpticalFlowTracking(cv::Mat previous, cv::Mat current, std::vector<cv::Point2f> &previous_pts, std::vector<cv::Point2f> &current_pts, std::vector<int> &trackIds)
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
                    trackIds.erase (trackIds.begin() + i - indexCorrection);
                    indexCorrection++;
        }

    }

    int indexCorrection_ = 0;
    int ptsNum = previous_pts.size();
    for(int i = 0; i < ptsNum; i++){
        double diff_x = std::fabs(previous_pts[i - indexCorrection_].x - current_pts[i - indexCorrection_].x);
        double diff_y = std::fabs(previous_pts[i - indexCorrection_].y - current_pts[i - indexCorrection_].y);
        // std::cout << diff_x << "  " << diff_y << std::endl;
        if(diff_x > 200 || diff_y > 120){
            previous_pts.erase ( previous_pts.begin() + i - indexCorrection_);
            current_pts.erase (current_pts.begin() + i - indexCorrection_);
            indexCorrection_++;           
        }
    }   
}

cv::Mat DrawKLTmatchLine(cv::Mat image1, cv::Mat image2, std::vector<cv::Point2f> previous_pts, std::vector<cv::Point2f> current_pts)
{
    cv::Mat MatchImg; 
    cv::hconcat(image1, image2, MatchImg);
    const int rNum = image1.cols;
    std::vector<cv::Point2f> rImgPtsf(current_pts.size());
    for(int i = 0; i < current_pts.size(); i++){
        cv::Point2f pts(current_pts[i].x + rNum, current_pts[i].y);
        rImgPtsf[i] = pts;
    }
    if (MatchImg.channels() < 3) cv::cvtColor(MatchImg, MatchImg, cv::COLOR_GRAY2RGB);
    for(int i = 0; i < previous_pts.size(); i++){
        cv::line(MatchImg, previous_pts[i], rImgPtsf[i], cv::Scalar(0,255,0), 1);
        cv::circle(MatchImg, previous_pts[i], 3, cv::Scalar(255,0, 0), 1);
        cv::circle(MatchImg, rImgPtsf[i], 3, cv::Scalar(255,0, 0), 1);
    }

    return MatchImg.clone();
}

cv::Mat DrawKLTmatchLine_vertical(cv::Mat image1, cv::Mat image2, std::vector<cv::Point2f> previous_pts, std::vector<cv::Point2f> current_pts)
{
    cv::Mat MatchImg; 
    cv::vconcat(image1, image2, MatchImg);
    const int rNum = image1.rows;
    std::vector<cv::Point2f> rImgPtsf(current_pts.size());
    for(int i = 0; i < current_pts.size(); i++){
        cv::Point2f pts(current_pts[i].x , current_pts[i].y + rNum);
        rImgPtsf[i] = pts;
    }
    if (MatchImg.channels() < 3) cv::cvtColor(MatchImg, MatchImg, cv::COLOR_GRAY2RGB);
    for(int i = 0; i < previous_pts.size(); i++){
        cv::line(MatchImg, previous_pts[i], rImgPtsf[i], cv::Scalar(0,255,0), 1);
        cv::circle(MatchImg, previous_pts[i], 3, cv::Scalar(255,0, 0), 1);
        cv::circle(MatchImg, rImgPtsf[i], 3, cv::Scalar(255,0, 0), 1);
    }

    return MatchImg.clone();
}

void RemoveMPoutlier(std::vector<cv::Point3d> &mp, std::vector<cv::Point2f> &lpts, std::vector<cv::Point2f> &rpts, const Vector6d pose)
{
    Eigen::Matrix<double, 4, 1> initCamView;
    initCamView << 0, 0, 1, 0;
    Eigen::Matrix4d camPose = To44RT(pose);
    
    Eigen::Matrix<double, 4, 1> currCamView_ = camPose * initCamView;
    std::cout << "current view : " << currCamView_.transpose() << std::endl;
    Eigen::Vector3d currCamView;
    currCamView << currCamView_(0), currCamView_(1), currCamView_(2);
    
    // std::vector<cv::Point3d> clone_map_point(mp);
    int mpNum = mp.size();
    int indexCorrection = 0;
        for(int i = 0; i < mpNum; i++){

            Eigen::Vector3d mpView;
            mpView.x() = mp[i - indexCorrection].x - pose[3];
            mpView.y() = mp[i - indexCorrection].y - pose[4];
            mpView.z() = mp[i - indexCorrection].z - pose[5];

                    
            // Eigen::Vector4d vecMP;
            // vecMP << clone_map_point[i].x, clone_map_point[i].y, clone_map_point[i].z, 0;

            if(currCamView.dot(mpView) < 0 || mpView.dot(mpView) > 100)
            {
                mp.erase(mp.begin() + i - indexCorrection);
                lpts.erase(lpts.begin() + i - indexCorrection);
                rpts.erase(rpts.begin() + i - indexCorrection);
                indexCorrection++;
            }
                
        }

    std::cout << " remove cam back and remain landmark num :  " << mp.size() << std::endl;    
    int indexCorrection_ = 0;
    Eigen::Matrix4d Pose44 = To44RT(pose);
    std::vector<float> reProjErr = ReprojectionError(mp, lpts, Pose44);
    for(int i = 0; i < reProjErr.size(); i++){
        if(reProjErr[i] > 3.5){
            mp.erase(mp.begin() + i - indexCorrection_);
            lpts.erase(lpts.begin() + i - indexCorrection_);
            rpts.erase(rpts.begin() + i - indexCorrection_);
        }
    }
}
                        
void RemoveOutlierMatch(std::vector<cv::Point2f> &lpts, std::vector<cv::Point2f> &rpts)
{
    cv::Mat inlierMask;
    cv::Mat E = cv::findEssentialMat(lpts, rpts, fx, c, cv::RANSAC, 0.999, 1, inlierMask);

    std::vector<cv::Point2f> clone_lpts(lpts);
    std::vector<cv::Point2f> clone_rpts(rpts);
    lpts.clear();
    rpts.clear();
    for(int i = 0; i < inlierMask.rows; i++){
      if(inlierMask.at<bool>(i, 0) == 1){
        lpts.push_back(clone_lpts[i]);
        rpts.push_back(clone_rpts[i]);
      }
    }    
}