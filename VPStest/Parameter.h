#pragma once

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"    
    
    
// Camera Parameter
// 00-02
float fx(718.856), fy(718.856), cx(607.193), cy(185.216);
float data[] = {fx, 0.0,cx, 
                0.0, fy, cy,
                0.0, 0.0, 0.0};       
cv::Mat K(3, 3, CV_32F, data);