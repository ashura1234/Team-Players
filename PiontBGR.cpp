#include "PointBGR.h"

PointBGR::PointBGR(int id, cv::Mat features) {
    dimensions = features.rows;
    pointId = id;
    for (int i = 0; i < dimensions; i++) {
        cv::Vec3d floatVec;
        floatVec[0] = features.at<cv::Vec3f>(i, 0)[0];
        floatVec[1] = features.at<cv::Vec3f>(i, 0)[1];
        floatVec[2] = features.at<cv::Vec3f>(i, 0)[2];
        this->features.push_back(floatVec);
    }
    clusterId = 0; //Initially not assigned to any cluster
}

int PointBGR::getDimensions() {
    return dimensions;
}

int PointBGR::getCluster() {
    return clusterId;
}

int PointBGR::getID() {
    return pointId;
}

void PointBGR::setCluster(int val) {
    clusterId = val;
}

cv::Vec3d PointBGR::getVal(int pos) {
    return this->features[pos];
}