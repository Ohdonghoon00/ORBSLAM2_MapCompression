#pragma once
#include "GL/freeglut.h" 
// #include <GL/gl.h>
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>
// #include "../types/Map.h"
// #include "../types/common.h"





void initialize_window();


// void initialize_window_for_BA();


void show_trajectory(const GLdouble &x, const GLdouble &y, const GLdouble &z, const double r, const double g, const double b, const double size);


void show_trajectory_keyframe(Eigen::Matrix4d rt, const double r, const double g, const double b, const double LineSize, const float TriangleSize);


// void show_trajectory_left_keyframe_mini(const GLdouble &x, const GLdouble &y, const GLdouble &z, const double r, const double g, const double b, const double size);


// void show_trajectory_right_mini(const GLdouble &x, const GLdouble &y, const GLdouble &z, const double r, const double g, const double b, const double size);

    
// void show_trajectory_left_mini(const GLdouble &x, const GLdouble &y, const GLdouble &z, const double r, const double g, const double b, const double size);


// void show_loop_detect_line(Map &a, int loop_detect_keyframe_id, int curr_keyframe_n, const float r, const float g, const float b, const double size);