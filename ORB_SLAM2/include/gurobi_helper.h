#pragma once

#include "gurobi_c++.h"
#include <vector>
#include <Eigen/Dense>
#include "algorithm"
#include "Map.h"
#include "DataBase.h"
#include "System.h"

class DataBase;

std::vector<GRBVar> CreateVariablesBinaryVector(int PointCloudNum, GRBModel& model_);

Eigen::Matrix<double, Eigen::Dynamic, 1> CalculateObservationCountWeight(ORB_SLAM2::Map* map_data);
Eigen::Matrix<double, Eigen::Dynamic, 1> CalculateObservationCountWeight2(DataBase* DB);

void SetObjectiveILP(std::vector<GRBVar> x_, Eigen::Matrix<double, Eigen::Dynamic, 1> q_, GRBModel& model_);

Eigen::MatrixXd CalculateVisibilityMatrix(ORB_SLAM2::Map* map_data);
Eigen::MatrixXd CalculateVisibilityMatrix2(DataBase* DB);

void AddConstraint(ORB_SLAM2::Map* map_data, GRBModel& model_, Eigen::MatrixXd A, std::vector<GRBVar> x, double CompressionRatio);
void AddConstraint2(DataBase* DB, GRBModel& model_, Eigen::MatrixXd A, std::vector<GRBVar> x, double CompressionRatio);
