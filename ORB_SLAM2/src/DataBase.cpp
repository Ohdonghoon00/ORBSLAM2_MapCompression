#include "DataBase.h"



namespace ORB_SLAM2
{

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

}
template void DataBase::serialize(boost::archive::binary_iarchive&, const unsigned int);
template void DataBase::serialize(boost::archive::binary_oarchive&, const unsigned int);

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

}