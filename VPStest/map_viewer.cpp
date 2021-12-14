
#include "map_viewer.h"


void initialize_window()
{
    int mode = GLUT_RGB | GLUT_SINGLE;
    glutInitDisplayMode(mode);              // Set drawing surface property
    glutInitWindowPosition(0, 0);       // Set window Position at Screen
    glutInitWindowSize(1000,1000);          // Set window size. Set printed working area size. Bigger than this size
    glutCreateWindow("VPStest_Viewer");         // Generate window. argument is window's name

    glClearColor(1.0, 1.0, 1.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT); 
}

// void initialize_window_for_BA()
// {
//     int mode = GLUT_RGB | GLUT_SINGLE;
//     glutInitDisplayMode(mode);              // Set drawing surface property
//     glutInitWindowPosition(1000, 0);       // Set window Position at Screen
//     glutInitWindowSize(500,500);          // Set window size. Set printed working area size. Bigger than this size
//     glutCreateWindow("GT and after BA trajectory");         // Generate window. argument is window's name

//     glClearColor(1.0, 1.0, 1.0, 0.0);
//     glClear(GL_COLOR_BUFFER_BIT); 
// }

void show_trajectory(const GLdouble &x, const GLdouble &y, const GLdouble &z, const double r, const double g, const double b, const double size)
{
    glColor3f(r,g,b);
    glPointSize(size);
    glBegin(GL_POINTS);
    glVertex3d(x*0.0025, z*0.0025 - 0.35, y*-0.0025);
    glEnd();
}

void show_trajectory_keyframe(Eigen::Matrix4f rt, const double r, const double g, const double b, const double LineSize, const float TriangleSize)
{
    glColor3f(r,g,b);
    glLineWidth(LineSize);
    glBegin(GL_LINE_LOOP);
    Eigen::Matrix<float, 3, 3> R = rt.block<3, 3>(0, 0);
    Eigen::Matrix<float, 3, 1> T = rt.block<3, 1>(0, 3);
    Eigen::Matrix<float, 3, 3> B;
    B <<    0, -TriangleSize, TriangleSize,
            0,             0,            0,
            0, -TriangleSize, -TriangleSize; 

    Eigen::Matrix<float, 3, 3> rb = R * B;
    rb.col(0) = rb.col(0) + T;
    rb.col(1) = rb.col(1) + T;
    rb.col(2) = rb.col(2) + T;

    for(int i = 0 ; i < 3; i++)
    {
        GLdouble x(rb(0, i)), y(rb(1, i)), z(rb(2, i));
        glVertex3d(x*0.0025, z*0.0025 - 0.35, y*-0.0025);
    }    
    
    // glVertex3d(x*0.003, z*0.003 - 0.8, y*-0.003);
    // glVertex3d(x*0.003, z*0.003 - 0.8, y*-0.003);
    glEnd();
}

// void show_trajectory_left_keyframe_mini(const GLdouble &x, const GLdouble &y, const GLdouble &z, const double r, const double g, const double b, const double size)
// {
//     glColor3f(r,g,b);
//     glLineWidth(size);
//     glBegin(GL_LINE_LOOP);
//     glVertex3d(x*0.001 - 0.01 - 0.5, z*0.001 - 0.01 + 0.5, y*-0.001 - 0.01);
//     glVertex3d(x*0.001 + 0.01 - 0.5, z*0.001 - 0.01 + 0.5, y*-0.001 - 0.01);
//     glVertex3d(x*0.001 - 0.5, z*0.001 + 0.01 + 0.5, y*-0.001);
//     glEnd();
// }

// void show_trajectory_right_mini(const GLdouble &x, const GLdouble &y, const GLdouble &z, const double r, const double g, const double b, const double size)
// {
//     glColor3f(r,g,b);
//     glPointSize(size);
//     glBegin(GL_POINTS);
//     glVertex3d(x*0.0007 + 0.5, z*0.0007+ 0.4, y*-0.0007);
//     glEnd();
// }
    
// void show_trajectory_left_mini(const GLdouble &x, const GLdouble &y, const GLdouble &z, const double r, const double g, const double b, const double size)
// {
//     glColor3f(r,g,b);
//     glPointSize(size);
//     glBegin(GL_POINTS);
//     glVertex3d(x*0.0007 - 0.5, z*0.0007 + 0.4, y*-0.0007);
//     glEnd();
// }

// void show_loop_detect_line(Map &a, int loop_detect_keyframe_id, int curr_keyframe_n, const float r, const float g, const float b, const double size)
// {
    
//     GLdouble x_loop = vec6d_to_homogenous_campose(a.keyframe[loop_detect_keyframe_id].cam_pose).at<double>(0, 3);
//     GLdouble y_loop = vec6d_to_homogenous_campose(a.keyframe[loop_detect_keyframe_id].cam_pose).at<double>(1, 3);
//     GLdouble z_loop = vec6d_to_homogenous_campose(a.keyframe[loop_detect_keyframe_id].cam_pose).at<double>(2, 3);
    
//     GLdouble x_curr = vec6d_to_homogenous_campose(a.keyframe[curr_keyframe_n].cam_pose).at<double>(0, 3);
//     GLdouble y_curr = vec6d_to_homogenous_campose(a.keyframe[curr_keyframe_n].cam_pose).at<double>(1, 3);
//     GLdouble z_curr = vec6d_to_homogenous_campose(a.keyframe[curr_keyframe_n].cam_pose).at<double>(2, 3);

//     glColor3f(r,g,b);
//     glLineWidth(size);
//     glBegin(GL_LINES);
//     glVertex3d(x_loop*0.005, z_loop*0.0011 - 0.3, y_loop*-0.0011);
//     glVertex3d(x_curr*0.0011, z_curr*0.0011 - 0.3, y_curr*-0.0011);
//     glEnd();
// }