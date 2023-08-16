//
// Created by tzuk on 6/18/23.
//

#ifndef STATISTICS_LINEARREGRESSION_H
#define STATISTICS_LINEARREGRESSION_H

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>

class LinearRegression {
public:
    LinearRegression(std::vector<double> &x, std::vector<double> &y);
    LinearRegression();
    void run();

    const std::vector<double> &getResiduals() const {
        return residuals;
    }
    void setX(std::vector<double> &x);
    void setY(std::vector<double> &y);
    // Getter for meanX
    double getMeanX() const {
        return meanX;
    }

    // Getter for meanY
    double getMeanY() const {
        return meanY;
    }

    // Getter for coVariance
    double getCoVariance() const {
        return coVariance;
    }

    // Getter for xVariance
    double getXVariance() const {
        return xVariance;
    }

    // Getter for yVariance
    double getYVariance() const {
        return yVariance;
    }

    // Getter for linearCorrelation
    double getLinearCorrelation() const {
        return linearCorrelation;
    }

    // Getter for slope
    double getSlope() const {
        return slope;
    }

    // Getter for yIntercepet
    double getYIntercept() const {
        return yIntercepet;
    }

    // Getter for SSE
    double getSSE() const {
        return SSE;
    }

    // Getter for SSR
    double getSSR() const {
        return SSR;
    }

    // Getter for SST
    double getSST() const {
        return SST;
    }

    // Getter for maxResidual
    double getMaxResidual() const {
        return maxResidual;
    }

    // Getter for rsquared
    double getRSquared() const {
        return rsquared;
    }

    // Getter for minResidual
    double getMinResidual() const {
        return minResidual;
    }

    // Getter for meanResidual
    double getMeanResidual() const {
        return meanResidual;
    }

private:
    void mean();

    void calculateLinearCorrelation();

    void calculateLinearEquation();

    void testResult();


    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> residuals;
    unsigned long size;
    double meanX;
    double meanY;
    double coVariance;
    double xVariance;
    double yVariance;
    double linearCorrelation;
    double fisherLinearCorrelation;
    double slope;
    double yIntercepet;
    double SSE;
    double SSR;
    double SST;
    double maxResidual;
    double rsquared;
    double minResidual;
    double meanResidual;
};


#endif //STATISTICS_LINEARREGRESSION_H
