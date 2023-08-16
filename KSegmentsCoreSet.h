//
// Created by tzuk on 4/17/23.
//

#ifndef ACTIVATIONPOTENTIALASPATTERNS_KSEGMENTSCORESET_H
#define ACTIVATIONPOTENTIALASPATTERNS_KSEGMENTSCORESET_H

#endif // ACTIVATIONPOTENTIALASPATTERNS_KSEGMENTSCORESET_H

#include <thread>
#include <vector>
#include <cmath>
#include <Eigen/SVD>
#include <boost/histogram.hpp>
#include <numeric>
#include "LinearRegression.h"
#include "RobustLinearRegression.h"

void resetThreads(std::vector<std::thread> &threads, int amountOfThreads) {
    for (auto &thread: threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    threads = std::vector<std::thread>(amountOfThreads);
}

std::vector<double>
biCriteriaRobustLinearRegression(std::vector<double> P, int sizeOfPattern, bool calculateDataDivision = true,
                                 int amountOfThreads = 1) {
    int segmentSize = sizeOfPattern;
    if (calculateDataDivision) {
        segmentSize = floor(P.size() / sizeOfPattern);
    }
    std::vector<std::vector<Eigen::Vector2d>> segments(P.size() / segmentSize);
    for (int i = 0; i < P.size(); i += segmentSize) {
        auto seg = std::vector<double>{P.begin() + i, P.begin() + i + segmentSize};
        segments[i / segmentSize] = std::vector<Eigen::Vector2d>{};
        for (int j = 0; j < seg.size(); ++j) {
            segments[i / segmentSize].emplace_back(j, seg[j]);
        }
    }
    std::vector<std::thread> threads(amountOfThreads);
    std::vector<double> directions(segments.size());
    int currentThread = 0;
    for (int i = 0; i < segments.size(); i++) {
        std::thread t([&](int index) {
                          RobustLinearRegression robustLinearRegression(segments[index], 20, 0.1, 1000);
                          auto result = robustLinearRegression.fit();
                          directions[index] = result.second;
                      },
                      i);
        threads[currentThread++] = std::move(t);
        if (currentThread >= amountOfThreads) {
            resetThreads(threads, amountOfThreads);
            currentThread = 0;
        }
    }
    for (auto &thread: threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    return directions;
}

std::vector<double>
biCriteriaLinerRegression(std::vector<double> P, int sizeOfPattern, bool calculateDataDivision = true,
                          int amountOfThreads = 1) {
    int segmentSize = sizeOfPattern;
    if (calculateDataDivision) {
        segmentSize = floor(P.size() / sizeOfPattern);
    }
    std::vector<std::vector<double>> segments(P.size() / segmentSize);
    std::vector<std::vector<double>> x(P.size() / segmentSize);
    for (int i = 0; i < P.size(); i += segmentSize) {
        segments[i / segmentSize] = std::vector<double>{P.begin() + i, P.begin() + i + segmentSize};
        x[i / segmentSize] = std::vector<double>(segments[i / segmentSize].size());
        std::iota(x[i / segmentSize].begin(), x[i / segmentSize].end(), 0);
    }
    std::vector<std::thread> threads(amountOfThreads);
    std::vector<double> directions(segments.size());
    int currentThread = 0;
    for (int i = 0; i < segments.size(); i++) {
        std::thread t([&](int index) {
                          LinearRegression slr(x[index], segments[index]);
                          slr.run();
                          double dir = slr.getSlope();
                          directions[index] = dir;
                      },
                      i);
        threads[currentThread++] = std::move(t);
        if (currentThread >= amountOfThreads) {
            resetThreads(threads, amountOfThreads);
            currentThread = 0;
        }
    }
    for (auto &thread: threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    return {directions};
}

std::vector<double>
biCriteriaSVD(std::vector<double> P, int sizeOfPattern, bool calculateDataDivision = true, int amountOfThreads = 1) {
    int segmentSize = sizeOfPattern;
    if (calculateDataDivision) {
        segmentSize = floor(P.size() / sizeOfPattern);
    }
    std::vector<Eigen::MatrixXd> segments(P.size() / segmentSize);
    for (int i = 0; i < P.size(); i += segmentSize) {
        auto seg = std::vector<double>{P.begin() + i, P.begin() + i + segmentSize};
        segments[i / segmentSize] = Eigen::MatrixXd(segmentSize, 2);
        double avg_x = 0.0;
        double avg_y = 0.0;
        for (int j = 0; j < seg.size(); ++j) {
            avg_x += j;
            avg_y += seg[j];
        }
        avg_x /= seg.size();
        avg_y /= seg.size();
        for (int j = 0; j < seg.size(); ++j) {
            segments[i / segmentSize](j, 0) = j - avg_x;
            segments[i / segmentSize](j, 1) = seg[j] - avg_y;
        }
    }
    std::vector<std::thread> threads(amountOfThreads);
    std::vector<double> directions(segments.size());
    int currentThread = 0;
    for (int i = 0; i < segments.size(); i++) {
        std::thread t([&](int index) {
                          Eigen::JacobiSVD<Eigen::MatrixXd> svd(segments[index].normalized(),
                                                                Eigen::ComputeThinU | Eigen::ComputeThinV);
                          svd.computeV();
                          auto v = svd.matrixV().col(0);
                          double dir = atan2(v(1), v(0));
                          directions[index] = dir;
                      },
                      i);
        threads[currentThread++] = std::move(t);
        if (currentThread >= amountOfThreads) {
            resetThreads(threads, amountOfThreads);
            currentThread = 0;
        }
    }
    for (auto &thread: threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    return directions;
}