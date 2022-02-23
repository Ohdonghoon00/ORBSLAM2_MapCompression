#include "VPStestResult.h"

template<class Archive>
void VPStestResult::serialize(Archive &ar, const unsigned int version)
{
        ar & MatchingImg;
        // ar & PnPpose;
        ar & PnPInlierRatio;
        ar & PnPInliers;
        ar & DBoW2ResultImgNum;
        // ar & Inliers;
        // ar & ReprojectionErr;
        // ar & KFimageNum;

}
template void VPStestResult::serialize(boost::archive::binary_iarchive&, const unsigned int);
template void VPStestResult::serialize(boost::archive::binary_oarchive&, const unsigned int);