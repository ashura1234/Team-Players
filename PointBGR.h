#pragma once
#include <vector>
#include <opencv2/opencv.hpp> 
#include <opencv2/core/types_c.h>

class PointBGR {

private:
    int pointId, clusterId;
    int dimensions;
    std::vector<cv::Vec3d> features;

public:
    PointBGR(int id, cv::Mat features);
    int getDimensions();
    int getCluster();
    int getID();
    void setCluster(int val);
    cv::Vec3d getVal(int pos);
};