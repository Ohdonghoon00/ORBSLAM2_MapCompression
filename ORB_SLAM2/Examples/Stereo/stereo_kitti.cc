/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <cstdio>

#include <Eigen/Dense>

#include<opencv2/core/core.hpp>
#include "Converter.h"
#include "VPStest.h"
#include<System.h>
#include "DataBase.h"
#include <ctime>
#include "BoostArchiver.h"

using namespace std;
using namespace DBoW2;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::STEREO,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;   

    // Main loop
    cv::Mat imLeft, imRight;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the images to the SLAM system
        SLAM.TrackStereo(imLeft,imRight,tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Map Compresssion
    // SLAM.MapCompression();
    
    
    
    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    
    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

    
    // Map Compression
    SLAM.MapCompression();










    // DataBase.h
    std::cout << " Save DataBase " << std::endl;
    std::vector<ORB_SLAM2::KeyFrame*> AllKFptr = SLAM.mpMap->GetAllKeyFrames();
    std::vector<ORB_SLAM2::MapPoint*> AllMpptr = SLAM.mpMap->GetAllMapPoints();
    std::sort(AllKFptr.begin(), AllKFptr.end(), ORB_SLAM2::KeyFrame::lId);
    
    DataBase DB;
    std::cout << " Keyframe num : " << AllKFptr.size() << std::endl;
    std::cout << " Landmarks num : " << AllMpptr.size() << std::endl;

    // Storage Map info
    for(size_t i = 0; i < AllMpptr.size(); i++){
        DB.Landmarks[i].x = (AllMpptr[i]->GetWorldPos()).at<float>(0, 0);
        DB.Landmarks[i].y = (AllMpptr[i]->GetWorldPos()).at<float>(1, 0);
        DB.Landmarks[i].z = (AllMpptr[i]->GetWorldPos()).at<float>(2, 0);
        DB.Descriptors.push_back(AllMpptr[i]->GetDescriptor());
    }

    // Storage KF info
    for(size_t i = 0; i < AllKFptr.size(); i++){
        for(size_t j = 0; j < AllMpptr.size(); j++){
            bool isinKF = AllMpptr[j]->IsInKeyFrame(AllKFptr[i]);
            if(isinKF) DB.KFtoMPIdx[i].push_back(j);
        }
        DB.timestamps.push_back(AllKFptr[i]->mTimeStamp);

    }

    

    // Save DataBase.h
    std::ofstream out("Kitti00_DB_b30_30%.bin", std::ios_base::binary);
    if (!out)
    {
        std::cout << "Cannot Write to Database File: "  << std::endl;
        exit(-1);
    }
    boost::archive::binary_oarchive oa(out, boost::archive::no_header);
    oa << DB;
    out.close();

    // Load DataBase.h
    // std::ifstream in("Kitti00_DB_original.bin", std::ios_base::binary);
    // if (!in)
    // {
    //     std::cout << "Cannot  You need create it first!" << std::endl;
    //     return false;
    // }
    // boost::archive::binary_iarchive ia(in, boost::archive::no_header);
    // ia >> DB;
    // in.close();

    // Save Timestamp + Trajectory Result 
    std::cout << " Save timestamp + trajectory " << std::endl;
    ofstream file;
    file.open("Kitti00_DB_b30_30%_result.txt");
    for(size_t i = 0; i < AllKFptr.size(); i++){
        cv::Mat cam_pose = AllKFptr[i]->GetPose();
        cam_pose = cam_pose.inv();
        // cv::Mat translation = AllKFptr[i]->GetTranslation();
        // cv::Mat rotation = AllKFptr[i]->GetRotation();
        
        Eigen::Matrix3f rot;
        rot <<  cam_pose.at<float>(0, 0), cam_pose.at<float>(0, 1), cam_pose.at<float>(0, 2),
                cam_pose.at<float>(1, 0), cam_pose.at<float>(1, 1), cam_pose.at<float>(1, 2),
                cam_pose.at<float>(2, 0), cam_pose.at<float>(2, 1), cam_pose.at<float>(2, 2);
        Eigen::Quaternionf q(rot);

        file << AllKFptr[i]-> mTimeStamp << " " << 
                cam_pose.at<float>(0, 3) << " " << cam_pose.at<float>(1, 3) << " " << cam_pose.at<float>(2, 3) << " " <<
                q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }

     file.close();


std::cout << std::endl;

    // Visualize CompressionMap
    // DrawMapCompression(CpMp);
    std::cout << " Finish Map Compression " << std::endl;
    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    // const int nTimes = 100;

    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }


}
            

    



    


        



    


    

