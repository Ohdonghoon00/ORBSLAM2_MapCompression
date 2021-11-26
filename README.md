# ORBSLAM2_MapCompression



## Compress stereo SLAM
- `ORB_SLAM2/Examples/Stereo/stereo_kitti.cc`



## Usage
### Compressed Parameter
- fix parameter b and compression ratio
in `ORB_SLAM2/src/gurobi_helper.cpp`

### output result
- fix path in `ORB_SLAM2/Examples/Stereo/stereo_kitti.cc`

- Ex) Kitti 00 dataset
```
cd ~/ORBSLAM2_MapCompression/ORB_SLAM2/
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI00-02.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/00
```

## Reference
https://github.com/raulmur/ORB_SLAM2


# VPStest
- Evaluate Compressed Map

## Usage

### output result
- fix path in `VPStest/main.cpp` 

- Ex/ Kitti 00 dataset
```
cd ~/ORBSLAM2_MapCompression/VPStest/build/
./VPStest INPUT_DATABASE.bin ../Vocabulary/ORBvoc.txt PATH_TO_DATASET_FOLDER/dataset/sequences/00
```

