#pragma once

#include "gurobi_c++.h"
#include <vector>
#include <Eigen/Dense>
#include "algorithm"
#include "Map.h"

std::vector<GRBVar> CreateVariablesBinaryVector(int PointCloudNum, GRBModel& model_);

Eigen::Matrix<double, Eigen::Dynamic, 1> CalculateObservationCountWeight(ORB_SLAM2::Map* map_data);

void SetObjectiveILP(std::vector<GRBVar> x_, Eigen::Matrix<double, Eigen::Dynamic, 1> q_, GRBModel& model_);

Eigen::MatrixXd CalculateVisibilityMatrix(ORB_SLAM2::Map* map_data);

void AddConstraint(ORB_SLAM2::Map* map_data, GRBModel& model_, Eigen::MatrixXd A, std::vector<GRBVar> x, double CompressionRatio);
