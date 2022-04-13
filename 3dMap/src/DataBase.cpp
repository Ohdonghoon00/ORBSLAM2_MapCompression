#include "DataBase.h"





template<class Archive>
void DataBase::serialize(Archive &ar, const unsigned int version)
{
        ar & Landmarks;
        ar & Descriptors;
        ar & KFtoMPIdx;
        ar & timestamps;
        ar & KeyPointInMap;
        ar & LeftKFimg;
        ar & RightKFimg;
        ar & kfPoses;
        ar & dbow2Descriptors;

}
template void DataBase::serialize(boost::archive::binary_iarchive&, const unsigned int);
template void DataBase::serialize(boost::archive::binary_oarchive&, const unsigned int);

void DataBase::SaveResultToDB(DataBase *db, Map map, std::vector<Keyframe> kf)
{
    for(int i = 0; i < kf.size(); i++){
        
        db->LeftKFimg.push_back(kf[i].limage);
        db->RightKFimg.push_back(kf[i].rimage);
        
        db->KeyPointInMap[i] = Converter::Point2f2KeyPoint(kf[i].lkeypoint);
        db->KFtoMPIdx[i].assign(kf[i].lptsIds.begin(), kf[i].lptsIds.end());
        db->timestamps.push_back(kf[i].timeStamp);
        db->kfPoses[i] = kf[i].camPose;

    }
}

cv::Mat DataBase::GetKFMatDescriptor(int idx)
{
    cv::Mat KFDescriptor = Descriptors[KFtoMPIdx[idx].front()];
    for(size_t i = 0; i < KFtoMPIdx[idx].size() - 1; i++){
        
        cv::Mat Descriptor;
        Descriptor = Descriptors[KFtoMPIdx[idx][i + 1]];
        cv::vconcat(KFDescriptor, Descriptor, KFDescriptor);
    }
    
    return KFDescriptor;
}    

std::vector<cv::Point3d> DataBase::GetKF3dPoint(int idx)
{
    std::vector<cv::Point3d> KFLandmark;
    KFLandmark.resize(KFtoMPIdx[idx].size());
    for(size_t i = 0; i < KFtoMPIdx[idx].size(); i++){
        cv::Point3d landmark3dpoint(Landmarks[KFtoMPIdx[idx][i]]); 
        KFLandmark[i] = landmark3dpoint;
    }

    return KFLandmark;
}

std::vector<cv::KeyPoint> DataBase::GetKF2dPoint(int idx)
{
    std::vector< cv::KeyPoint > KF2dPoints = KeyPointInMap[idx];
    return KF2dPoints;
}

cv::Mat DataBase::GetNearReferenceKFMatDescriptor(int rkidx, int nearRkN)
{
    std::set<int> PointIdx;
    for(int i = -nearRkN + rkidx; i < nearRkN + rkidx + 1; i++){
        if(i < 0) continue;   
        for(size_t j = 0; j < KFtoMPIdx[i].size(); j++){
            PointIdx.insert(KFtoMPIdx[i][j]);
        }
    }
   
    int count = 0;
    cv::Mat KFDescriptor = Descriptors[*PointIdx.begin()];
    for(auto i : PointIdx){
        if(count == 0){
            count++;
            continue;
        }
        
        cv::Mat Descriptor;
        Descriptor = Descriptors[i];
        cv::vconcat(KFDescriptor, Descriptor, KFDescriptor);
    }
        
    return KFDescriptor;
} 

std::vector<cv::Point3f> DataBase::GetNearReferenceKF3dPoint(int rkidx, int nearRkN)
{
    std::set<int> PointIdx;
    for(int i = -nearRkN + rkidx; i < nearRkN + rkidx + 1; i++){
        
        if(i < 0) continue;
        for(size_t j = 0; j < KFtoMPIdx[i].size(); j++)
            PointIdx.insert(KFtoMPIdx[i][j]);    
    }
    
    
    std::vector<cv::Point3f> KFLandmark;
    for(auto i : PointIdx){
        cv::Point3f landmark3dpoint(Landmarks[i]); 
        KFLandmark.push_back(landmark3dpoint);       
    }

    return KFLandmark;
}

int DataBase::GetObservationCount(int idx)
{
    int count = 0;
    for(int i = 0; i < KFtoMPIdx.size(); i++){
        auto it = std::find(KFtoMPIdx[i].begin(), KFtoMPIdx[i].end(), idx);
        if(it != KFtoMPIdx[i].end())
            count++;
    }

    return count;
}