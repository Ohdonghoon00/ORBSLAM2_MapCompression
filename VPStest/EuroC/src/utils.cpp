#include "utils.h"

//////////////////////////////////////////////////////////
Eigen::Vector3d Origin{0.0, 0.0, 0.0};
Eigen::Vector3d ZVec = Eigen::Vector3d::UnitZ();
Eigen::Matrix3d Iden = Eigen::Matrix3d::Identity();

//////////// EuroC ////////////////////

float fx(435.2046959714599), fy(435.2046959714599), cx(367.4517211914062), cy(252.2008514404297);
float IntrinsicData[] = {   fx, 0.0,cx, 
                            0.0, fy, cy,
                            0.0, 0.0, 1.0}; 


///// Extrinsic /////
// Machine Hall  body(IMU) - cam0 //
double Cam2BodyData[] = {0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0};





// ViconRoom1  body(IMU) - cam0 //



// ViconRoom2  body(IMU) - cam0 //






//////////////////////////////////////////////////////////

cv::Mat GetK(float* IntrinsicData)
{
    cv::Mat K(3, 3, CV_32F, IntrinsicData);
    return K;
}

Eigen::Matrix4d GetCam2Body(double * Cam2BodyData)
{
    Eigen::Matrix4d Cam2Body = Eigen::Map<Eigen::Matrix4d>(Cam2BodyData);
    return Cam2Body.transpose();
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

Vector6d To6DOF(cv::Mat R, cv::Mat T)
{


    Eigen::Matrix4d Pose;
    cv::Rodrigues(R, R);
    Pose << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), T.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), T.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), T.at<double>(2, 0),
            0, 0, 0, 1;

    
    Vector6d Pose_ = To6DOF(Pose);
    return Pose_;
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

Vector6d ProjectionTo6DOFPoses(cv::Mat R, cv::Mat T)
{
    Eigen::Matrix4d Pose;
    cv::Rodrigues(R, R);
    Pose << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), T.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), T.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), T.at<double>(2, 0),
            0, 0, 0, 1;
    Pose = Pose.inverse();
    Vector6d Pose_ = To6DOF(Pose);
    return Pose_;   
}

Vector6d ProjectionTo6DOFPoses(Vector6d proj)
{
    Eigen::Matrix4d Proj_ = To44RT(proj);
    Eigen::Matrix4d Pose = Proj_.inverse();
    Vector6d Pose_ = To6DOF(Pose);
    return Pose_;   
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