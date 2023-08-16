//
// Created by tzuk on 7/24/23.
//

#include "RobustLinearRegression.h"

std::pair<double, double> RobustLinearRegression::fit() {
    // Initializing parameters
    double alpha = 0;  // intercept
    double beta = 0;   // slope


    for (int i = 0; i < maxIterations; i++) {
        double alpha_gradient = 0;
        double beta_gradient = 0;

        for (auto &point: data) {
            double prediction = alpha + beta * point.x();
            double residual = point.y() - prediction;

            alpha_gradient += derivativeHuber(residual);
            beta_gradient += derivativeHuber(residual) * point.x();
        }

        // Update parameters using gradient descent
        alpha += learningRate * alpha_gradient / double(data.size());
        beta += learningRate * beta_gradient / double(data.size());
    }

    return std::make_pair(alpha, beta);
}

double RobustLinearRegression::derivativeHuber(double residual) {
    if (std::abs(residual) <= delta) {
        return residual;
    } else {
        return delta * ((residual > 0) ? 1 : -1);
    }
}

double RobustLinearRegression::huberLoss(double residual) {
    if (std::abs(residual) <= delta) {
        return 0.5 * residual * residual;
    } else {
        return delta * (std::abs(residual) - 0.5 * delta);
    }
}

RobustLinearRegression::RobustLinearRegression(std::vector<Eigen::Vector2d> &inputData, double deltaVal,
                                               double learningRate,int maxIterations) : data(inputData), delta(deltaVal),
                                                                      learningRate(learningRate),maxIterations(maxIterations) {}

