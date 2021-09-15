#pragma once
#include <vector>
#include "PointBGR.h"
#include <opencv2/opencv.hpp> 
#include <opencv2/core/types_c.h>

class Cluster {

private:
    int clusterId;
    std::vector<cv::Vec3d> centroid;
    std::vector<PointBGR> points;

public:
    Cluster(int clusterId, PointBGR centroid);
    void addPoint(PointBGR p);
    bool removePoint(int pointId);
    int getId();
    PointBGR getPoint(int pos);
    int getSize();
    cv::Vec3d getCentroidByPos(int pos);
    void setCentroidByPos(int pos, cv::Vec3d val);
};