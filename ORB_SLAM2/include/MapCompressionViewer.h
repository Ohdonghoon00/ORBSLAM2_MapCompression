#pragma once

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Tracking.h"
#include "System.h"
#include"Map.h"
#include"MapPoint.h"
#include"KeyFrame.h"
#include "Viewer.h"
#include<pangolin/pangolin.h>



void DrawMapCompression(std::vector<cv::Mat> AllMp);
void DrawCompressionMapPoints(std::vector<cv::Mat> AllMp);
// void DrawCompressionMapPoints();
