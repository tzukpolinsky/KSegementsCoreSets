//
// Created by tzuk on 7/24/23.
//

#ifndef ACTIVATIONPOTENTIALASPATTERNS_ROBUSTLINEARREGRESSION_H
#define ACTIVATIONPOTENTIALASPATTERNS_ROBUSTLINEARREGRESSION_H


#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Eigen>

class RobustLinearRegression {
private:
    std::vector<Eigen::Vector2d> data;
    double delta;
    double learningRate = 0.03;

    double huberLoss(double residual);

    double derivativeHuber(double residual);

    int maxIterations;
public:
    RobustLinearRegression(std::vector<Eigen::Vector2d> &inputData, double deltaVal, double learningRate,
                           int maxIterations = 500);

    std::pair<double, double> fit();
};

#endif //ACTIVATIONPOTENTIALASPATTERNS_ROBUSTLINEARREGRESSION_H
