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

#include "System.h"
#include "Converter.h"
#include "gurobi_helper.h"
#include "DataBase.h"
#include "BoostArchiver.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>

namespace ORB_SLAM2
{

    System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
                   const bool bUseViewer) : mSensor(sensor), mpViewer(static_cast<Viewer *>(NULL)), mbReset(false), mbActivateLocalizationMode(false),
                                            mbDeactivateLocalizationMode(false)
    {
        // Output welcome message
        cout << endl
             << "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl
             << "This program comes with ABSOLUTELY NO WARRANTY;" << endl
             << "This is free software, and you are welcome to redistribute it" << endl
             << "under certain conditions. See LICENSE.txt." << endl
             << endl;

        cout << "Input sensor was set to: ";

        if (mSensor == MONOCULAR)
            cout << "Monocular" << endl;
        else if (mSensor == STEREO)
            cout << "Stereo" << endl;
        else if (mSensor == RGBD)
            cout << "RGB-D" << endl;

        // Check settings file
        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if (!fsSettings.isOpened())
        {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }

        // Load ORB Vocabulary
        cout << endl
             << "Loading ORB Vocabulary. This could take a while..." << endl;

        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if (!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl
             << endl;

        // Create DataBase ( Save DataBase.h After SLAM )
        OriginalDB = new DataBase();
        

        // Create KeyFrame Database
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        // Create the Map
        mpMap = new Map();

        // Create Drawers. These are used by the Viewer
        mpFrameDrawer = new FrameDrawer(mpMap);
        mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

        // Initialize the Tracking thread
        //(it will live in the main thread of execution, the one that called this constructor)
        mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                                 mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

        // Initialize the Local Mapping thread and launch
        mpLocalMapper = new LocalMapping(mpMap, mSensor == MONOCULAR);
        mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run, mpLocalMapper);

        // Initialize the Loop Closing thread and launch
        mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor != MONOCULAR);
        mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

        // Initialize the Viewer thread and launch
        if (bUseViewer)
        {
            mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, strSettingsFile);
            mptViewer = new thread(&Viewer::Run, mpViewer);
            mpTracker->SetViewer(mpViewer);
        }

        // Set pointers between threads
        mpTracker->SetLocalMapper(mpLocalMapper);
        mpTracker->SetLoopClosing(mpLoopCloser);

        mpLocalMapper->SetTracker(mpTracker);
        mpLocalMapper->SetLoopCloser(mpLoopCloser);

        mpLoopCloser->SetTracker(mpTracker);
        mpLoopCloser->SetLocalMapper(mpLocalMapper);
    }

    cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
    {
        if (mSensor != STEREO)
        {
            cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
            exit(-1);
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if (mbActivateLocalizationMode)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped())
                {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if (mbDeactivateLocalizationMode)
            {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if (mbReset)
            {
                mpTracker->Reset();
                mbReset = false;
            }
        }

        cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft, imRight, timestamp);

        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
        return Tcw;
    }

    cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
    {
        if (mSensor != RGBD)
        {
            cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
            exit(-1);
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if (mbActivateLocalizationMode)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped())
                {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if (mbDeactivateLocalizationMode)
            {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if (mbReset)
            {
                mpTracker->Reset();
                mbReset = false;
            }
        }

        cv::Mat Tcw = mpTracker->GrabImageRGBD(im, depthmap, timestamp);

        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
        return Tcw;
    }

    cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
    {
        if (mSensor != MONOCULAR)
        {
            cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
            exit(-1);
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if (mbActivateLocalizationMode)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped())
                {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if (mbDeactivateLocalizationMode)
            {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if (mbReset)
            {
                mpTracker->Reset();
                mbReset = false;
            }
        }

        cv::Mat Tcw = mpTracker->GrabImageMonocular(im, timestamp);

        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

        return Tcw;
    }

    void System::ActivateLocalizationMode()
    {
        unique_lock<mutex> lock(mMutexMode);
        mbActivateLocalizationMode = true;
    }

    void System::DeactivateLocalizationMode()
    {
        unique_lock<mutex> lock(mMutexMode);
        mbDeactivateLocalizationMode = true;
    }

    bool System::MapChanged()
    {
        static int n = 0;
        int curn = mpMap->GetLastBigChangeIdx();
        if (n < curn)
        {
            n = curn;
            return true;
        }
        else
            return false;
    }

    void System::Reset()
    {
        unique_lock<mutex> lock(mMutexReset);
        mbReset = true;
    }

    void System::Shutdown()
    {
        mpLocalMapper->RequestFinish();
        mpLoopCloser->RequestFinish();
        if (mpViewer)
        {
            mpViewer->RequestFinish();
            while (!mpViewer->isFinished())
                usleep(5000);
        }

        // Wait until all thread have effectively stopped
        while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
        {
            usleep(5000);
        }

        if (mpViewer)
            pangolin::BindToContext("ORB-SLAM2: Map Viewer");
    }

    void System::SaveTrajectoryTUM(const string &filename)
    {
        cout << endl
             << "Saving camera trajectory to " << filename << " ..." << endl;
        if (mSensor == MONOCULAR)
        {
            cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
            return;
        }

        vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
        // We need to get first the keyframe pose and then concatenate the relative transformation.
        // Frames not localized (tracking failure) are not saved.

        // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
        // which is true when tracking failed (lbL).
        list<ORB_SLAM2::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
        list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
        list<bool>::iterator lbL = mpTracker->mlbLost.begin();
        for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
                                     lend = mpTracker->mlRelativeFramePoses.end();
             lit != lend; lit++, lRit++, lT++, lbL++)
        {
            if (*lbL)
                continue;

            KeyFrame *pKF = *lRit;

            cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

            // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
            while (pKF->isBad())
            {
                Trw = Trw * pKF->mTcp;
                pKF = pKF->GetParent();
            }

            Trw = Trw * pKF->GetPose() * Two;

            cv::Mat Tcw = (*lit) * Trw;
            cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

            vector<float> q = Converter::toQuaternion(Rwc);

            f << setprecision(9) << *lT << " " << setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        }
        f.close();
        cout << endl
             << "trajectory saved!" << endl;
    }

    void System::SaveKeyFrameTrajectoryTUM(const string &filename)
    {
        cout << endl
             << "Saving keyframe trajectory to " << filename << " ..." << endl;

        vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        // cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];

            // pKF->SetPose(pKF->GetPose()*Two);

            if (pKF->isBad())
                continue;

            cv::Mat R = pKF->GetRotation().t();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat t = pKF->GetCameraCenter();
            f << setprecision(9) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
              << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        }

        f.close();
        cout << endl
             << "trajectory saved!" << endl;
    }

    void System::SaveTrajectoryKITTI(const string &filename)
    {
        cout << endl
             << "Saving camera trajectory to " << filename << " ..." << endl;
        if (mSensor == MONOCULAR)
        {
            cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
            return;
        }

        vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
        // We need to get first the keyframe pose and then concatenate the relative transformation.
        // Frames not localized (tracking failure) are not saved.

        // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
        // which is true when tracking failed (lbL).
        list<ORB_SLAM2::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
        list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
        for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++)
        {
            ORB_SLAM2::KeyFrame *pKF = *lRit;

            cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

            while (pKF->isBad())
            {
                //  cout << "bad parent" << endl;
                Trw = Trw * pKF->mTcp;
                pKF = pKF->GetParent();
            }

            Trw = Trw * pKF->GetPose() * Two;

            cv::Mat Tcw = (*lit) * Trw;
            cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

            f << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2) << " " << twc.at<float>(0) << " " << Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " " << twc.at<float>(1) << " " << Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " " << twc.at<float>(2) << endl;
        }
        f.close();
        cout << endl
             << "trajectory saved!" << endl;
    }

    int System::GetTrackingState()
    {
        unique_lock<mutex> lock(mMutexState);
        return mTrackingState;
    }

    vector<MapPoint *> System::GetTrackedMapPoints()
    {
        unique_lock<mutex> lock(mMutexState);
        return mTrackedMapPoints;
    }

    vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
    {
        unique_lock<mutex> lock(mMutexState);
        return mTrackedKeyPointsUn;
    }

    void System::MapCompression(double CompressionRatio)
    {
        // Map Compression
        std::cout << "Map Compression ... " << std::endl;
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);

        long unsigned int PointCloudNum = mpMap->MapPointsInMap();

        std::cout << " Create Variables ... " << std::endl;
        // Create Variables
        std::vector<GRBVar> x = CreateVariablesBinaryVector(PointCloudNum, model);

        std::cout << " Set Objective ... " << std::endl;
        // Set Objective
        Eigen::Matrix<double, Eigen::Dynamic, 1> q = CalculateObservationCountWeight(mpMap);
        SetObjectiveILP(x, q, model);

        std::cout << " Add Constraint ... " << std::endl;
        // Add Constraint
        Eigen::MatrixXd A = CalculateVisibilityMatrix(mpMap);
        AddConstraint(mpMap, model, A, x, CompressionRatio);

        std::cout << std::endl;

        std::cout << " Optimize model ... " << std::endl;
        // Optimize model
        model.optimize();

        std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << endl;

        std::cout << std::endl;

        // Erase Map Point
        size_t index = 0;
        for (size_t i = 0; i < x.size(); i++)
        {

            if (x[i].get(GRB_DoubleAttr_X) == 0)
            {
                mpMap->EraseMapPoint((mpMap->GetAllMapPoints())[i - index]);
                index++;
            }
        }
        std::cout << " Finish Map Compression" << std::endl;
    }

    void System::SaveCompressedDataBase(std::vector<GRBVar> x, std::string filepath)
    {
        CompressedDB = new DataBase();
        *CompressedDB = *OriginalDB;
        std::cout << "Original Landmark num : " << x.size() << std::endl;
        int index = 0;
        for(size_t i = 0; i < x.size(); i++){
            
            if (x[i].get(GRB_DoubleAttr_X) == 0){
                
                CompressedDB->Landmarks.erase(i);
                CompressedDB->Descriptors.erase(i);
                index++;
                
                for(int j = 0; j < CompressedDB->KFtoMPIdx.size(); j++){
                    
                    auto it = std::find(CompressedDB->KFtoMPIdx[j].begin(), CompressedDB->KFtoMPIdx[j].end(), i);
                    if(it == CompressedDB->KFtoMPIdx[j].end())
                        ;
                    else{
                        int idx = it - CompressedDB->KFtoMPIdx[j].begin();
                        CompressedDB->KFtoMPIdx[j].erase(CompressedDB->KFtoMPIdx[j].begin() + idx);
                        CompressedDB->KeyPointInMap[j].erase(CompressedDB->KeyPointInMap[j].begin() + idx);
                    }
                    
                }
            }
        }
        // std::map< int, cv::Point3f > Landmarks_(CompressedDB->Landmarks);
        // CompressedDB->Landmarks.clear();
        // std::map<int, cv::Point3f>::iterator iter;
        // int cnt = 0;
        // for(iter = Landmarks_.begin(); iter != Landmarks_.end(); iter++){
        //     CompressedDB->Landmarks[cnt++] = iter->second;
        // }

        std::cout << "Compressed landmark num : " << CompressedDB->Landmarks.size() << std::endl;
        std::cout << "Compressed landmark num : " << CompressedDB->Descriptors.size() << std::endl;
        // Saved DataBase to Binary file
        std::ofstream out(filepath, std::ios_base::binary);
        if (!out)
        {
            std::cout << "Cannot Write to Database File: " << std::endl;
            exit(-1);
        }
        boost::archive::binary_oarchive oa(out, boost::archive::no_header);
        oa << CompressedDB;
        out.close();
    } 

    void System::MapCompression2(double CompressionRatio, std::string filepath)
    {
        try{
        // Map Compression
        std::cout << "Map Compression ... " << std::endl;
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);

        long unsigned int PointCloudNum = OriginalDB->Landmarks.size();
        std::cout << "original landmark num : " << PointCloudNum << std::endl;
        std::cout << " Create Variables ... " << std::endl;
        // Create Variables
        std::vector<GRBVar> x = CreateVariablesBinaryVector(PointCloudNum, model);

        std::cout << " Set Objective ... " << std::endl;
        // Set Objective
        Eigen::Matrix<double, Eigen::Dynamic, 1> q = CalculateObservationCountWeight2(OriginalDB);
        SetObjectiveILP(x, q, model);

        std::cout << " Add Constraint ... " << std::endl;
        // Add Constraint
        Eigen::MatrixXd A = CalculateVisibilityMatrix2(OriginalDB);
        AddConstraint2(OriginalDB, model, A, x, CompressionRatio);

        std::cout << std::endl;

        std::cout << " Optimize model ... " << std::endl;
        // Optimize model
        model.optimize();

        int optimstatus = model.get(GRB_IntAttr_Status);
        cout << "Optimization complete" << endl;
        double objval = 0;
        if (optimstatus == GRB_OPTIMAL) {
        objval = model.get(GRB_DoubleAttr_ObjVal);
        cout << "Optimal objective: " << objval << endl;
        } else if (optimstatus == GRB_INF_OR_UNBD) {
        cout << "Model is infeasible or unbounded" << endl;
        } else if (optimstatus == GRB_INFEASIBLE) {
        cout << "Model is infeasible" << endl;
        } else if (optimstatus == GRB_UNBOUNDED) {
        cout << "Model is unbounded" << endl;
        } else {
        cout << "Optimization was stopped with status = "
            << optimstatus << endl;
        }

        // std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
        SaveCompressedDataBase(x, filepath);
        } catch(GRBException e){
            std::cout << "Error code : " << e.getErrorCode() << std::endl;
            std::cout << " Message: " << e.getMessage() << std::endl;
        } catch(...){
            std::cout << "Exceptioon during optimization" << std::endl;
        }

        std::cout << std::endl;

        // Erase Map Point

        std::cout << " Finish Map Compression" << std::endl;
    }

    void System::SaveOriginalDataBase(std::string filepath)
    {

        // DataBase.h
        std::cout << " Save DataBase " << std::endl;
        std::vector<ORB_SLAM2::KeyFrame *> AllKFptr = mpMap->GetAllKeyFrames();
        std::vector<ORB_SLAM2::MapPoint *> AllMpptr = mpMap->GetAllMapPoints();
        std::sort(AllKFptr.begin(), AllKFptr.end(), ORB_SLAM2::KeyFrame::lId);

        int MaxMapId = 0;
        for (size_t i = 0; i < AllKFptr.size(); i++)
        {
            OriginalDB->timestamps.push_back(AllKFptr[i]->mTimeStamp);
            OriginalDB->LeftKFimg.push_back(AllKFptr[i]->LeftImg);
            OriginalDB->RightKFimg.push_back(AllKFptr[i]->RightImg);
            
            // std::cout <<  DB->KeyPointInMap[i].size() << "  " << DB->KFtoMPIdx[i].size() << std::endl;
            std::set<MapPoint*> KFMapPoints = AllKFptr[i]->GetMapPoints();
            // std::cout << "KF num : " << i << " map points num : " << KFMapPoints.size() << std::endl;
            for(auto j : KFMapPoints){

                
                int KeypointIdx = j->GetIndexInKeyFrame(AllKFptr[i]);
                cv::KeyPoint Point2d = AllKFptr[i]->mvKeys[KeypointIdx];
                OriginalDB->KeyPointInMap[i].push_back(Point2d);
                
                cv::Point3f mp;
                mp.x = (j->GetWorldPos()).at<float>(0, 0);
                mp.y = (j->GetWorldPos()).at<float>(1, 0);
                mp.z = (j->GetWorldPos()).at<float>(2, 0);

                bool InKF = false;
                int Mpid = 0;
                for(size_t k = 0; k < OriginalDB->Landmarks.size(); k++){
                    if(mp == OriginalDB->Landmarks[k]){
                        InKF = true;
                        Mpid = k;
                    }
                }
                     

                if(InKF){
                    // std::cout << "Already In DB " << std::endl;
                    OriginalDB->KFtoMPIdx[i].push_back(Mpid);
                }
                else{
                    OriginalDB->Landmarks[MaxMapId] = mp;
                    OriginalDB->Descriptors[MaxMapId] = j->GetDescriptor();
                    OriginalDB->KFtoMPIdx[i].push_back(MaxMapId);
                    MaxMapId++;
                }
            }
        }
        // for(size_t i = 0; i < OriginalDB->KFtoMPIdx.size(); i++){
        //     std::cout << "KF num : " << i << std::endl;
        //     for(size_t j = 0; j < OriginalDB->KFtoMPIdx[i].size(); j++){
        //         std::cout << OriginalDB->KFtoMPIdx[i][j] << " ";

        //     }
        //     std::cout << std::endl;
        // }
        std::cout << "  DB landmark num : " << OriginalDB->Landmarks.size() << std::endl;
        std::cout << OriginalDB->GetObservationCount(5) << std::endl;
        // Saved DataBase to Binary file
        std::ofstream out(filepath, std::ios_base::binary);
        if (!out)
        {
            std::cout << "Cannot Write to Database File: " << std::endl;
            exit(-1);
        }
        boost::archive::binary_oarchive oa(out, boost::archive::no_header);
        oa << OriginalDB;
        out.close();
    }

       
               
        


    void System::SavePose(std::string filepath)
    {
        std::cout << " Save timestamp + trajectory " << std::endl;
        ofstream file;
        std::vector<ORB_SLAM2::KeyFrame *> AllKFptr = mpMap->GetAllKeyFrames();

        std::sort(AllKFptr.begin(), AllKFptr.end(), ORB_SLAM2::KeyFrame::lId);

        file.open(filepath);
        for (size_t i = 0; i < AllKFptr.size(); i++)
        {

            cv::Mat cam_pose = AllKFptr[i]->GetPose();
            cam_pose = cam_pose.inv();

            Eigen::Matrix3f rot;
            rot << cam_pose.at<float>(0, 0), cam_pose.at<float>(0, 1), cam_pose.at<float>(0, 2),
                cam_pose.at<float>(1, 0), cam_pose.at<float>(1, 1), cam_pose.at<float>(1, 2),
                cam_pose.at<float>(2, 0), cam_pose.at<float>(2, 1), cam_pose.at<float>(2, 2);
            Eigen::Quaternionf q(rot);

            file << setprecision(19) << AllKFptr[i]->mTimeStamp << " " << cam_pose.at<float>(0, 3) << " " << cam_pose.at<float>(1, 3) << " " << cam_pose.at<float>(2, 3) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }

        file.close();
    }

    void System::printMap(std::map<ORB_SLAM2::KeyFrame *, size_t> target_map)
    {
        for (std::map<ORB_SLAM2::KeyFrame *, size_t>::iterator Iter = target_map.begin(); Iter != target_map.end(); ++Iter)
            std::cout << Iter->first->mnId << "   " << Iter->second << std::endl;
    }

} // namespace ORB_SLAM
