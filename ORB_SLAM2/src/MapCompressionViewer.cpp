

#include "MapCompressionViewer.h"
#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "System.h"
#include <pangolin/pangolin.h>
#include <ctime>


// 00 - 02
// float mKeyFrameSize(0.6f);
// float mKeyFrameLineWidth(2.f);
// float mGraphLineWidth(1.f);
// float mPointSize(2.f);
// float mCameraSize(0.7f);
// float mCameraLineWidth(3.f);
// float mViewpointX(0.f), mViewpointY(-100.f), mViewpointZ(-0.1f), mViewpointF(2000.f);

// 04 - 12
float mKeyFrameSize(0.6f);
float mKeyFrameLineWidth(2.f);
float mGraphLineWidth(1.f);
float mPointSize(2.f);
float mCameraSize(0.7f);
float mCameraLineWidth(3.f);
float mViewpointX(0.f), mViewpointY(-100.f), mViewpointZ(-0.1f), mViewpointF(2000.f);

void DrawMapCompression(std::vector<cv::Mat> AllMp)
{


    pangolin::CreateWindowAndBind("ORB-SLAM2: ComPresssion Map Viewer",1024,768);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    // pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    // pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    // pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    // pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
    // pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);
    // pangolin::Var<bool> menuReset("menu.Reset",false,false);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    // cv::namedWindow("ORB-SLAM2: Current Frame");

    // bool bFollow = true;
    // bool bLocalizationMode = false;


    // bool ViewerStop = true;
    while(1)
    {
    //     int key = cv::waitKey(1);
    //     if (key == 13) break;
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

        // if(menuFollowCamera && bFollow)
        // {
        //     s_cam.Follow(Twc);
        // }
        // else if(menuFollowCamera && !bFollow)
        // {
        //     s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
        //     s_cam.Follow(Twc);
        //     bFollow = true;
        // }
        // else if(!menuFollowCamera && bFollow)
        // {
        //     bFollow = false;
        // }

        // if(menuLocalizationMode && !bLocalizationMode)
        // {
        //     mpSystem->ActivateLocalizationMode();
        //     bLocalizationMode = true;
        // }
        // else if(!menuLocalizationMode && bLocalizationMode)
        // {
        //     mpSystem->DeactivateLocalizationMode();
        //     bLocalizationMode = false;
        // }

        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        // mpMapDrawer->DrawCurrentCamera(Twc);
        // if(menuShowKeyFrames || menuShowGraph)
            // mpMapDrawer->DrawCompressionKeyFrames();
        // if(menuShowPoints)
            // mpMapDrawer->DrawCompressionMapPoints();
        DrawCompressionMapPoints(AllMp);
        pangolin::FinishFrame();
    }
        // cv::Mat im = mpFrameDrawer->DrawFrame();
        // cv::imshow("ORB-SLAM2: Current Frame",im);
        // cv::waitKey(mT);

        // if(menuReset)
        // {
        //     menuShowGraph = true;
        //     menuShowKeyFrames = true;
        //     menuShowPoints = true;
        //     menuLocalizationMode = false;
        //     if(bLocalizationMode)
        //         mpSystem->DeactivateLocalizationMode();
        //     bLocalizationMode = false;
        //     bFollow = true;
        //     menuFollowCamera = true;
        //     mpSystem->Reset();
        //     menuReset = false;
        // }

    //     if(Stop())
    //     {
    //         while(isStopped())
    //         {
    //             usleep(3000);
    //         }
    //     }

    //     if(CheckFinish())
    //         break;
    // }

    // SetFinish();
}

// void DrawCompressionKeyFrames()
// {
//     const float &w = mKeyFrameSize;
//     const float h = w*0.75;
//     const float z = w*0.6;

//     const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();


//         for(size_t i=0; i<vpKFs.size(); i++)
//         {
//             KeyFrame* pKF = vpKFs[i];
//             cv::Mat Twc = pKF->GetPoseInverse().t();

//             glPushMatrix();

//             glMultMatrixf(Twc.ptr<GLfloat>(0));

//             glLineWidth(mKeyFrameLineWidth);
//             glColor3f(0.0f,0.0f,1.0f);
//             glBegin(GL_LINES);
//             glVertex3f(0,0,0);
//             glVertex3f(w,h,z);
//             glVertex3f(0,0,0);
//             glVertex3f(w,-h,z);
//             glVertex3f(0,0,0);
//             glVertex3f(-w,-h,z);
//             glVertex3f(0,0,0);
//             glVertex3f(-w,h,z);

//             glVertex3f(w,h,z);
//             glVertex3f(w,-h,z);

//             glVertex3f(-w,h,z);
//             glVertex3f(-w,-h,z);

//             glVertex3f(-w,h,z);
//             glVertex3f(w,h,z);

//             glVertex3f(-w,-h,z);
//             glVertex3f(w,-h,z);
//             glEnd();

//             glPopMatrix();
//         }

    
// }

void DrawCompressionMapPoints(std::vector<cv::Mat> AllMp)
{
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);

    for(size_t i = 0; i < AllMp.size(); i++)
    {
        // if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
        //     continue;
        // cv::Mat pos = AllMp[i]->GetWorldPos();
        glVertex3f(AllMp[i].at<float>(0),AllMp[i].at<float>(1),AllMp[i].at<float>(2));
    }
    glEnd();
    





}
