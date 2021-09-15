#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include "Player.h"
#include "PointBGR.h"
#include "Cluster.h"

class Kmeans_PointBGR {
private:
    int K, iters, dimensions, total_points;
    double weight;
    std::vector<Cluster> clusters;
    std::vector<PointBGR> all_points;
    double square(cv::Vec3d vec);
    int getNearestClusterId(PointBGR point);

public:
    Kmeans_PointBGR(int K, int iterations, double weight, std::vector<Player>& players);
    void run(std::vector<Player>& players);
};

/*int (std::vector<Player> players) {

    //Need 2 arguments (except filename) to run, else exit
    if (argc != 3) {
        cout << "Error: command-line argument count mismatch.";
        return 1;
    }

    //Fetching number of clusters
    int K = atoi(argv[2]);

    //Open file for fetching points
    string filename = argv[1];
    ifstream infile(filename.c_str());

    if (!infile.is_open()) {
        cout << "Error: Failed to open file." << endl;
        return 1;
    }

    //Fetching points from file
    int pointId = 1;
    vector<Point> all_points;
    string line;

    while (getline(infile, line)) {
        Point point(pointId, line);
        all_points.push_back(point);
        pointId++;
    }
    infile.close();
    cout << "\nData fetched successfully!" << endl << endl;

    //Return if number of clusters > number of points
    if (all_points.size() < K) {
        cout << "Error: Number of clusters greater than number of points." << endl;
        return 1;
    }

    //Running K-Means Clustering
    int iters = 100;

    KMeans kmeans(K, iters);
    kmeans.run(all_points);

    return 0;
}*/