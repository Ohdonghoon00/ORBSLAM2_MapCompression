#pragma once

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>
#include <cmath>

bool ISNaN(Eigen::VectorXf Es)
{
    bool Estimate = true;
    
    bool IsNan_x = isnan(Es(0));
    bool IsNan_y = isnan(Es(1));
    bool IsNan_z = isnan(Es(2));

    bool IsNan_qx = isnan(Es(3));
    bool IsNan_qy = isnan(Es(4));
    bool IsNan_qz = isnan(Es(5));
    bool IsNan_qw = isnan(Es(6));

    if(IsNan_x || IsNan_y || IsNan_z || IsNan_qx || IsNan_qy || IsNan_qz || IsNan_qw)
        Estimate = false;
    
    return Estimate;

}


int AbsoluteTrajectoryError(std::vector<Vector6d> GT, std::vector<Vector6d> Es, double &RMS_Trans_Error, double &RMS_Rot_Error)
{

    int count = 0;
    
    for(int i = 0; i < GT.size(); i++){
        
        // if(ISNaN(Es[i]) == false){
        //     // std::cout << Es[i](0) << std::endl;
        //     std::cout << count << std::endl;
            
        //     continue;
        // }
        Eigen::Matrix4d GTMotion = To44RT(GT[i]);
        Eigen::Matrix4d EsMotion = To44RT(Es[i]);
        Eigen::Matrix4d RelativePose = GTMotion.inverse() * EsMotion;

        // trans
        Eigen::Vector3d RelativeTrans;
        RelativeTrans << RelativePose(0, 3), RelativePose(1, 3), RelativePose(2, 3);
        RMS_Trans_Error += RelativeTrans.dot(RelativeTrans);

        // rotation
        Eigen::Matrix3d RelativeRot_ = RelativePose.block<3, 3>(0, 0);
        Eigen::Vector3d RelativeRot = ToVec3(RelativeRot_);
        RMS_Rot_Error += RelativeRot.dot(RelativeRot);

        count++;
    }
    
    RMS_Trans_Error = std::sqrt(RMS_Trans_Error / count);
    RMS_Rot_Error = std::sqrt(RMS_Rot_Error / count);
    
    return 0;
}

float RelativePoseError(std::vector<Eigen::VectorXf> GT, std::vector<Eigen::VectorXf> Es)
{
    float RMS_Error(0.0);
    int count = 0;
    int lastindex(0), currindex(0);
    float Posethres = 0.5;

    for(int i = 0; i < GT.size(); i++){
        
        if(std::abs(GT[i](2) - Es[i](2)) < Posethres && std::abs(GT[i](0) - Es[i](0)) < Posethres && std::abs(GT[i](1) - Es[i](1)) < Posethres){
 
        if(ISNaN(Es[i]) == false){
            // std::cout << Es[i](0) << std::endl;
            // std::cout << count << std::endl;
            
            continue;
        }
        
            
        if(currindex == 0){
            lastindex = i;
            currindex = 1;
            continue;
        }    
        

        currindex = i;
        // std::cout << lastindex << "   " << currindex << std::endl;
        Eigen::Quaternionf GT_q_curr, GT_q_last;
        Eigen::Quaternionf Es_q_curr, Es_q_last;

        GT_q_curr.x() = GT[currindex](3);
        GT_q_curr.y() = GT[currindex](4);
        GT_q_curr.z() = GT[currindex](5);
        GT_q_curr.w() = GT[currindex](6);
        Eigen::Matrix3f GT_R_curr(GT_q_curr);

        GT_q_last.x() = GT[lastindex](3);
        GT_q_last.y() = GT[lastindex](4);
        GT_q_last.z() = GT[lastindex](5);
        GT_q_last.w() = GT[lastindex](6);
        Eigen::Matrix3f GT_R_last(GT_q_last);

        Eigen::Matrix4f GTMotion_curr;
        GTMotion_curr <<    GT_R_curr(0, 0), GT_R_curr(0, 1), GT_R_curr(0, 2), GT[currindex](0),
                            GT_R_curr(1, 0), GT_R_curr(1, 1), GT_R_curr(1, 2), GT[currindex](1),
                            GT_R_curr(2, 0), GT_R_curr(2, 1), GT_R_curr(2, 2), GT[currindex](2),
                            0, 0, 0, 1;

        Eigen::Matrix4f GTMotion_last;
        GTMotion_last <<    GT_R_last(0, 0), GT_R_last(0, 1), GT_R_last(0, 2), GT[lastindex](0),
                            GT_R_last(1, 0), GT_R_last(1, 1), GT_R_last(1, 2), GT[lastindex](1),
                            GT_R_last(2, 0), GT_R_last(2, 1), GT_R_last(2, 2), GT[lastindex](2),
                            0, 0, 0, 1;
        

        Es_q_curr.x() = Es[currindex](3);
        Es_q_curr.y() = Es[currindex](4);
        Es_q_curr.z() = Es[currindex](5);
        Es_q_curr.w() = Es[currindex](6);
        Eigen::Matrix3f Es_R_curr = Es_q_curr.normalized().toRotationMatrix();

        Es_q_last.x() = Es[lastindex](3);
        Es_q_last.y() = Es[lastindex](4);
        Es_q_last.z() = Es[lastindex](5);
        Es_q_last.w() = Es[lastindex](6);
        Eigen::Matrix3f Es_R_last = Es_q_last.normalized().toRotationMatrix();

        Eigen::Matrix4f EsMotion_curr;
        EsMotion_curr <<    Es_R_curr(0, 0), Es_R_curr(0, 1), Es_R_curr(0, 2), Es[currindex](0),
                            Es_R_curr(1, 0), Es_R_curr(1, 1), Es_R_curr(1, 2), Es[currindex](1),
                            Es_R_curr(2, 0), Es_R_curr(2, 1), Es_R_curr(2, 2), Es[currindex](2),
                            0, 0, 0, 1;

        Eigen::Matrix4f EsMotion_last;
        EsMotion_last <<    Es_R_last(0, 0), Es_R_last(0, 1), Es_R_last(0, 2), Es[lastindex](0),
                            Es_R_last(1, 0), Es_R_last(1, 1), Es_R_last(1, 2), Es[lastindex](1),
                            Es_R_last(2, 0), Es_R_last(2, 1), Es_R_last(2, 2), Es[lastindex](2),
                            0, 0, 0, 1;            
            
            
            
            
            
            
            
            Eigen::Matrix4f GT_RelativePose = GTMotion_last.inverse() * GTMotion_curr;
            Eigen::Matrix4f Es_RelativePose = EsMotion_last.inverse() * EsMotion_curr;
            Eigen::Matrix4f RelativePose = GT_RelativePose.inverse() * Es_RelativePose;
            Eigen::Vector3f RelativeTrans;
            RelativeTrans << RelativePose(0, 3), RelativePose(1, 3), RelativePose(2, 3);
            RMS_Error += RelativeTrans.dot(RelativeTrans);
        
            count++;

            lastindex = currindex;
        }  

    }
    std::cout << "RPE count : " << count << std::endl;
    float RMS_Error_ = std::sqrt(RMS_Error / count);

    return RMS_Error_;

}

void TranslationError(std::vector<Eigen::VectorXf> GT, std::vector<Eigen::VectorXf> Es, std::ofstream &filepath)
{
    int count = 0;
    for(int i = 0; i < GT.size(); i++){
        
        if(ISNaN(Es[i]) == false){
            std::cout << Es[i](0) << std::endl;
            std::cout << count << std::endl;
            
            continue;
        }
        Eigen::Quaternionf GT_q;
        Eigen::Quaternionf Es_q;

        GT_q.x() = GT[i](3);
        GT_q.y() = GT[i](4);
        GT_q.z() = GT[i](5);
        GT_q.w() = GT[i](6);
        Eigen::Matrix3f GT_R(GT_q);
        
        Es_q.x() = Es[i](3);
        Es_q.y() = Es[i](4);
        Es_q.z() = Es[i](5);
        Es_q.w() = Es[i](6);
        Eigen::Matrix3f Es_R(Es_q);

        Eigen::Matrix4f GTMotion;
        GTMotion << GT_R(0, 0), GT_R(0, 1), GT_R(0, 2), GT[i](0),
                    GT_R(1, 0), GT_R(1, 1), GT_R(1, 2), GT[i](1),
                    GT_R(2, 0), GT_R(2, 1), GT_R(2, 2), GT[i](2),
                    0, 0, 0, 1;
        
        Eigen::Matrix4f EsMotion;
        EsMotion << Es_R(0, 0), Es_R(0, 1), Es_R(0, 2), Es[i](0),
                    Es_R(1, 0), Es_R(1, 1), Es_R(1, 2), Es[i](1),
                    Es_R(2, 0), Es_R(2, 1), Es_R(2, 2), Es[i](2),
                    0, 0, 0, 1;        
    
            
        Eigen::Matrix4f RelativePose = GTMotion.inverse() * EsMotion;
        Eigen::Vector3f RelativeTrans;
        RelativeTrans << std::abs(RelativePose(0, 3)), std::abs(RelativePose(1, 3)), std::abs(RelativePose(2, 3));
        double XYZError = std::sqrt(RelativeTrans.dot(RelativeTrans));

        if(XYZError > 0.5) {
            std::cout << "start : "  << i << " ";
            count++;
        }
        filepath << XYZError << std::endl;
            
        


   

    }
    std::cout << (double)GT.size() << std::endl;
    std::cout << "success ratio : " << (double)count / (double)GT.size() << std::endl;
}

void TranslationError(std::vector<Eigen::VectorXf> GT, Eigen::Matrix4d Es, int num)
{
        Eigen::Quaternionf GT_q;

        GT_q.x() = GT[num](3);
        GT_q.y() = GT[num](4);
        GT_q.z() = GT[num](5);
        GT_q.w() = GT[num](6);
        Eigen::Matrix3f GT_R(GT_q);
        Eigen::Matrix4f GTMotion;
        GTMotion << GT_R(0, 0), GT_R(0, 1), GT_R(0, 2), GT[num](0),
                    GT_R(1, 0), GT_R(1, 1), GT_R(1, 2), GT[num](1),
                    GT_R(2, 0), GT_R(2, 1), GT_R(2, 2), GT[num](2),
                    0, 0, 0, 1;
        
        Eigen::Matrix4f EsMotion = Es.cast<float>();
      
    
            
        Eigen::Matrix4f RelativePose = GTMotion.inverse() * EsMotion;
        Eigen::Vector3f RelativeTrans;
        RelativeTrans << std::abs(RelativePose(0, 3)), std::abs(RelativePose(1, 3)), std::abs(RelativePose(2, 3));
        double XYZError = std::sqrt(RelativeTrans.dot(RelativeTrans));
        std::cout << " Error : " << XYZError << std::endl;     
}