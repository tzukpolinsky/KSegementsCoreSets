//
// Created by tzuk on 6/18/23.
//

#include "LinearRegression.h"

LinearRegression::LinearRegression(std::vector<double> &x, std::vector<double> &y) {
    if (x.size() != y.size()) {
        std::cout << "x and y must be with the same length" << std::endl;
    }
    this->x = x;
    this->y = y;
    size = x.size();
}

void LinearRegression::mean() {
    meanX = meanY = 0.0;
    for (int i = 0; i < size; ++i) {
        meanX += x[i];
        meanY += y[i];
    }
    meanX /= double(size);
    meanY /= double(size);
}

void LinearRegression::calculateLinearCorrelation() {
    xVariance = 0.0;
    yVariance = 0.0;
    coVariance = 0.0;
    for (int i = 0; i < size; ++i) {
        double xDiff = x[i] - meanX;
        double yDiff = y[i] - meanY;
        coVariance += xDiff * yDiff;
        xVariance += std::pow(xDiff, 2);
        yVariance += std::pow(xDiff, 2);
    }
    linearCorrelation = coVariance / std::sqrt(xVariance * yVariance); //Pearson linear correlation
    coVariance /= double(size);
    xVariance /= double(size);
    yVariance /= double(size);
    fisherLinearCorrelation = 0.5 * std::log((1 + linearCorrelation) / (1 - linearCorrelation));
}

void LinearRegression::calculateLinearEquation() {
    slope = linearCorrelation * (yVariance / xVariance);
    yIntercepet = meanY - slope * meanX; // y =mx+b

}

void LinearRegression::testResult() {
    residuals.reserve(size);
    meanResidual = 0.0;
    maxResidual = 0.0;
    minResidual = std::numeric_limits<double>::max();;
    for (int n = 0; n < size; n++) {
        double yhat = yIntercepet + (slope * x[n]);
        double residual = y[n] - yhat;
        SSE += pow((yhat - meanY), 2);
        SSR += pow(residual, 2);
        residuals[n] = residual;
        if (residuals[n] > maxResidual) {
            maxResidual = residuals[n];
        }
        if (residuals[n] < minResidual) {
            minResidual = residuals[n];
        }
        meanResidual += fabsf64(residuals[n]);
    }
    meanResidual /= double(size);
    SST = SSR + SSE;
    rsquared =
            SSE / SST;           //Can also be obtained by r ^ 2 for simple regression (i.e. 1 independent variable)

}

void LinearRegression::run() {
    mean();
    calculateLinearCorrelation();
    calculateLinearEquation();
    testResult();
}

LinearRegression::LinearRegression() {

}

void LinearRegression::setX(std::vector<double> &x) {
    this->x = x;
    if (!y.empty()) {
        if (x.size() != y.size()) {
            std::cout << "could not create liner regression on different lengthen data" << std::endl;
        } else {
            size = x.size();
        }
    }
}

void LinearRegression::setY(std::vector<double> &y) {
    this->y = y;
    if (!x.empty()) {
        if (x.size() != y.size()) {
            std::cout << "could not create liner regression on different lengthen data" << std::endl;
        } else {
            size = y.size();
        }
    }
}
