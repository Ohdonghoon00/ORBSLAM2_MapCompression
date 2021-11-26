#include "DataBase.h"




template<class Archive>
void DataBase::serialize(Archive &ar, const unsigned int version)
{
        ar & Landmarks;
        ar & Descriptors;
        ar & KFtoMPIdx;
        ar & timestamps;

}
template void DataBase::serialize(boost::archive::binary_iarchive&, const unsigned int);
template void DataBase::serialize(boost::archive::binary_oarchive&, const unsigned int);

cv::Mat DataBase::GetKFMatDescriptor(int idx)
{
    cv::Mat KFDescriptor = Descriptors[KFtoMPIdx[idx][0]];
    for(size_t i = 0; i < KFtoMPIdx[idx].size() - 1; i++){
        cv::Mat Descriptor;
        Descriptor = Descriptors[KFtoMPIdx[idx][i + 1]];
        cv::vconcat(KFDescriptor, Descriptor, KFDescriptor);
    }
        
    return KFDescriptor;
}    

std::vector<cv::Point3f> DataBase::GetKF3dPoint(int idx)
{
    std::vector<cv::Point3f> KFLandmark;
    KFLandmark.resize(KFtoMPIdx[idx].size());
    for(size_t i = 0; i < KFtoMPIdx[idx].size(); i++){
        cv::Point3f landmark3dpoint(Landmarks[KFtoMPIdx[idx][i]]); 
        KFLandmark[i] = landmark3dpoint;
    }

    return KFLandmark;
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

