#include "Kmeans_PointBGR.h"

double Kmeans_PointBGR::square(cv::Vec3d vec) {
    return vec.dot(vec);
}

int Kmeans_PointBGR::getNearestClusterId(PointBGR point) {
    double sum = 0, min_dist;
    int NearestClusterId;

    for (int i = 0; i < dimensions; i++)
    {
        sum += square(clusters[0].getCentroidByPos(i) - point.getVal(i)) * std::pow(this->weight, 1 / (i + 1));
    }

    min_dist = sqrt(sum);
    NearestClusterId = clusters[0].getId();

    for (int i = 1; i < K; i++)
    {
        double dist;
        sum = 0.0;

        for (int j = 0; j < dimensions; j++)
        {
            sum += square((clusters[i].getCentroidByPos(j) - point.getVal(j)) * std::pow(this->weight, 1 / (i + 1)));
        }

        dist = sqrt(sum);

        if (dist < min_dist)
        {
            min_dist = dist;
            NearestClusterId = clusters[i].getId();
        }
    }

    return NearestClusterId;
}

Kmeans_PointBGR::Kmeans_PointBGR(int K, int iterations, double weight, std::vector<Player>& players) {
    auto start = std::chrono::high_resolution_clock::now();
    cout << "Start clustering teams." << endl;
    this->K = K;
    this->iters = iterations;
    this->dimensions = players[0].GetDominantColors().rows;
    this->total_points = players.size();
    this->weight = weight;

    for (int i = 0; i < total_points; i++) {
        PointBGR point(i, players[i].GetDominantColors());
        this->all_points.push_back(point);
    }
    run(players);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "Finished clustering teams." << endl;
    cout << "--------------------(" << duration.count() / 1000 << "ms)" << endl;
}

void Kmeans_PointBGR::run(std::vector<Player>& players) {

    total_points = this->all_points.size();
    dimensions = this->all_points[0].getDimensions();


    //Initializing Clusters
    std::vector<int> used_pointIds;

    for (int i = 1; i <= K; i++)
    {
        while (true)
        {
            int index = rand() % total_points;

            if (std::find(used_pointIds.begin(), used_pointIds.end(), index) == used_pointIds.end())
            {
                used_pointIds.push_back(index);
                this->all_points[index].setCluster(i);
                Cluster cluster(i, this->all_points[index]);
                clusters.push_back(cluster);
                break;
            }
        }
    }
    //cout << "Clusters initialized = " << clusters.size() << endl << endl;


    //cout << "Running K-Means Clustering.." << endl;

    int iter = 1;
    while (true)
    {
        //cout << "Iter - " << iter << "/" << iters << endl;
        bool done = true;

        // Add all points to their nearest cluster
        for (int i = 0; i < total_points; i++)
        {
            int currentClusterId = this->all_points[i].getCluster();
            int nearestClusterId = getNearestClusterId(this->all_points[i]);

            if (currentClusterId != nearestClusterId)
            {
                if (currentClusterId != 0) {
                    for (int j = 0; j < K; j++) {
                        if (clusters[j].getId() == currentClusterId) {
                            clusters[j].removePoint(this->all_points[i].getID());
                        }
                    }
                }

                for (int j = 0; j < K; j++) {
                    if (clusters[j].getId() == nearestClusterId) {
                        clusters[j].addPoint(this->all_points[i]);
                    }
                }
                this->all_points[i].setCluster(nearestClusterId);
                done = false;
            }
        }

        // Recalculating the center of each cluster
        for (int i = 0; i < K; i++)
        {
            int ClusterSize = clusters[i].getSize();

            for (int j = 0; j < dimensions; j++)
            {
                cv::Vec3d sum(0, 0, 0);
                if (ClusterSize > 0)
                {
                    for (int p = 0; p < ClusterSize; p++)
                        sum += clusters[i].getPoint(p).getVal(j);
                    clusters[i].setCentroidByPos(j, sum / ClusterSize);
                }
            }
        }

        if (done || iter >= iters)
        {
            //cout << "Clustering completed in iteration : " << iter << endl << endl;
            break;
        }
        iter++;
    }


    //Print pointIds in each cluster
    for (int i = 0; i < K; i++) {
        //cout << "Points in cluster " << clusters[i].getId() << " : ";
        for (int j = 0; j < clusters[i].getSize(); j++) {
            int pointId = clusters[i].getPoint(j).getID();
            players[pointId].SetTeam(i);
            //cout << pointId << " ";
        }
        //cout << endl << endl;
    }
    //cout << "========================" << endl << endl;

    //Write cluster centers to file
    /*std::ofstream outfile;
    outfile.open("clusters.txt");
    if (outfile.is_open()) {
        for (int i = 0; i < K; i++) {
            cout << "Cluster " << clusters[i].getId() << " centroid : ";
            for (int j = 0; j < dimensions; j++) {
                cout << clusters[i].getCentroidByPos(j) << " ";     //Output to console
                outfile << clusters[i].getCentroidByPos(j) << " ";  //Output to file
            }
            cout << endl;
            outfile << endl;
        }
        outfile.close();
    }
    else {
        cout << "Error: Unable to write to clusters.txt";
    }*/

}