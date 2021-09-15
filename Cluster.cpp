#include "Cluster.h"

Cluster::Cluster(int clusterId, PointBGR centroid) {
    this->clusterId = clusterId;
    for (int i = 0; i < centroid.getDimensions(); i++) {
        this->centroid.push_back(centroid.getVal(i));
    }
    this->addPoint(centroid);
}

void Cluster::addPoint(PointBGR p) {
    p.setCluster(this->clusterId);
    points.push_back(p);
}

bool Cluster::removePoint(int pointId) {
    int size = points.size();

    for (int i = 0; i < size; i++)
    {
        if (points[i].getID() == pointId)
        {
            points.erase(points.begin() + i);
            return true;
        }
    }
    return false;
}

int Cluster::getId() {
    return clusterId;
}

PointBGR Cluster::getPoint(int pos) {
    return points[pos];
}

int Cluster::getSize() {
    return points.size();
}

cv::Vec3d Cluster::getCentroidByPos(int pos) {
    return centroid[pos];
}

void Cluster::setCentroidByPos(int pos, cv::Vec3d val) {
    this->centroid[pos] = val;
}