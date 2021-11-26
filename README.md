# ORBSLAM2_MapCompression



## Compress stereo SLAM
`ORB_SLAM2/Examples/Stereo/stereo_kitti.cc`



## Usage
### Compressed Parameter
fix parameter b and compression ratio
in `ORB_SLAM2/src/gurobi_helper.cpp`

### ouput result
fix path `ORB_SLAM2/Examples/Stereo/stereo_kitti.cc`

Ex) kitti 00 dataset
```
cd ~/ORBSLAM2_MapCompression/ORB_SLAM2/
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI00-02.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/00
```

## Reference
https://github.com/raulmur/ORB_SLAM2


# VPStest
Evaluate Compressed Map
