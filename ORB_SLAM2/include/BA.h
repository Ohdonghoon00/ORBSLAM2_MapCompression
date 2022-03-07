#pragma once
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <opencv2/opencv.hpp>

struct ReprojectionError
{
    ReprojectionError(const cv::Point2f& _x, double _f, const cv::Point2d& _c) : x(_x), f(_f), c(_c) {}

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        // X' = R*X + t
        T X[3];
        ceres::AngleAxisRotatePoint(camera, point, X);
        X[0] += camera[3];
        X[1] += camera[4];
        X[2] += camera[5];

        // x' = K*X'
        T x_p = f * X[0] / X[2] + c.x;
        T y_p = f * X[1] / X[2] + c.y;

        // residual = x - x'
        residuals[0] = T(x.x) - x_p;
        residuals[1] = T(x.y) - y_p;





        return true;
    }

    static ceres::CostFunction* create (const cv::Point2f& _x, double _f, const cv::Point2d& _c)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(new ReprojectionError(_x, _f, _c)));
    }

    private:
        const cv::Point2f x;
        const double f;
        const cv::Point2d c;

};

struct motion_only_ReprojectionError
{
    motion_only_ReprojectionError(const cv::Point2f& _x, const cv::Point3d& _y, double _f, const cv::Point2d& _c) : x(_x), y(_y), f(_f), c(_c) {}

    template <typename T>
    bool operator()(const T* const camera, T* residuals) const
    {

        
        // const cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, c.x, 0, f, c.y, 0, 0, 1);
        // cv::Point2d p;
        
        
        // cv::projectPoints(y, r, t, K, cv::noArray(), p);

        // X' = R*X + t
        T point[3];
        point[0] = T(y.x); 
        point[1] = T(y.y);
        point[2] = T(y.z);
        
        T X[3];
        ceres::AngleAxisRotatePoint(camera, point, X);
        X[0] += camera[3];
        X[1] += camera[4];
        X[2] += camera[5];

        // x' = K*X'
        T x_p = f * X[0] / X[2] + c.x;
        T y_p = f * X[1] / X[2] + c.y;

        // residual = x - x'
        residuals[0] = T(x.x) - x_p;
        residuals[1] = T(x.y) - y_p;





        return true;
    }

    static ceres::CostFunction* create (const cv::Point2f& _x, const cv::Point3d& _y, double _f, const cv::Point2d& _c)
    {
        return (new ceres::AutoDiffCostFunction<motion_only_ReprojectionError, 2, 6>(new motion_only_ReprojectionError(_x, _y, _f, _c)));
    }

    private:
        const cv::Point2f x;
        const cv::Point3d y;
        const double f;
        const cv::Point2d c;
};

struct map_point_only_ReprojectionError_scale
{
    map_point_only_ReprojectionError_scale(const cv::Point2f& _x, const cv::Vec6d& _camera, double _f, const cv::Point2d& _c, double _s) : x(_x), _camera(_camera), f(_f), c(_c), s(_s) {}

    template <typename T>
    bool operator()(const T* const point, T* residuals) const
    {

        


        // X' = R*X + t
        T camera[3];
        camera[0] = T(_camera[0]); 
        camera[1] = T(_camera[1]);
        camera[2] = T(_camera[2]);
        
        T X[3];
        ceres::AngleAxisRotatePoint(camera, point, X);
        X[0] += T(_camera[3]);
        X[1] += T(_camera[4]);
        X[2] += T(_camera[5]);

        // x' = K*X'
        T x_p = f * X[0] / X[2] + c.x;
        T y_p = f * X[1] / X[2] + c.y;

        x_p = x_p * (1/s);
        y_p = y_p * (1/s);

        // residual = x - x'
        residuals[0] = T(x.x) - x_p;
        residuals[1] = T(x.y) - y_p;





        return true;
    }

    static ceres::CostFunction* create (const cv::Point2f& _x, const cv::Vec6d& _camera, double _f, const cv::Point2d& _c, double _s)
    {
        return (new ceres::AutoDiffCostFunction<map_point_only_ReprojectionError_scale, 2, 3>(new map_point_only_ReprojectionError_scale(_x, _camera, _f, _c, _s)));
    }

    private:
        const cv::Point2f x;
        const cv::Vec6d _camera;
        const double f;
        const cv::Point2d c;
        const double s;
};

struct map_point_only_ReprojectionError
{
    map_point_only_ReprojectionError(const cv::Point2f& _x, const cv::Vec6d& _camera, double _f, const cv::Point2d& _c) : x(_x), _camera(_camera), f(_f), c(_c) {}

    template <typename T>
    bool operator()(const T* const point, T* residuals) const
    {

        


        // X' = R*X + t
        T camera[3];
        camera[0] = T(_camera[0]); 
        camera[1] = T(_camera[1]);
        camera[2] = T(_camera[2]);
        
        T X[3];
        ceres::AngleAxisRotatePoint(camera, point, X);
        X[0] += T(_camera[3]);
        X[1] += T(_camera[4]);
        X[2] += T(_camera[5]);

        // x' = K*X'
        T x_p = f * X[0] / X[2] + c.x;
        T y_p = f * X[1] / X[2] + c.y;


        // residual = x - x'
        residuals[0] = T(x.x) - x_p;
        residuals[1] = T(x.y) - y_p;





        return true;
    }

    static ceres::CostFunction* create (const cv::Point2f& _x, const cv::Vec6d& _camera, double _f, const cv::Point2d& _c)
    {
        return (new ceres::AutoDiffCostFunction<map_point_only_ReprojectionError, 2, 3>(new map_point_only_ReprojectionError(_x, _camera, _f, _c)));
    }

    private:
        const cv::Point2f x;
        const cv::Vec6d _camera;
        const double f;
        const cv::Point2d c;
};