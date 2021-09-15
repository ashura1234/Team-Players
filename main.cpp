#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include "Player.h"
#include "Kmeans_PointBGR.h"

using namespace std;
using namespace std::chrono;

// Helper function visualizes the point cloud.
void ViewPC(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
	pcl::visualization::CloudViewer viewer("cloud");
	viewer.showCloud(cloud);
	while (!viewer.wasStopped()) {
	}
}

// Loads data from txt file to a pcl point cloud.
bool LoadCloud(const string& filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	auto start = high_resolution_clock::now();
	cout << "Start loading " << filename << endl;
	ifstream fs;
	fs.open(filename.c_str(), ios::binary);
	if (!fs.is_open() || fs.fail())
	{
		PCL_ERROR("Could not open file '%s'! Error : %s\n", filename.c_str(), strerror(errno));
		fs.close();
		return (false);
	}

	string line;
	vector<string> st;

	while (!fs.eof())
	{
		getline(fs, line);
		// Ignore empty lines
		if (line.empty())
			continue;

		// Tokenize the line
		boost::trim(line);
		boost::split(st, line, boost::is_any_of("\t\r "), boost::token_compress_on);

		if (st.size() != 6)
			continue;

		pcl::PointXYZRGB point;
		point.x = float(atof(st[0].c_str()));
		point.y = float(atof(st[1].c_str()));
		point.z = float(atof(st[2].c_str()));
		point.r = uint8_t(atof(st[3].c_str()));
		point.g = uint8_t(atof(st[4].c_str()));
		point.b = uint8_t(atof(st[5].c_str()));
		cloud->push_back(point);
	}
	fs.close();
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Loaded " << cloud->size() << "points." << endl;
	cout << "--------------------(" << duration.count()/1000 << "ms)" << endl;
	return true;
}

// Times RBG values of every point in the point cloud by a factor in order to increase successful clustering rate. 
// See Result/Accuracy Optimization section in readme.
void BrightnessCorrection(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float alpha, float beta)
{
	auto start = high_resolution_clock::now();
	cout << "Start adjusting contrast." << endl;
	for (int i = 0; i < cloud->points.size(); i++)
	{
		// saturate_cast prevents values from being over 255
		cloud->points[i].r = cv::saturate_cast<unsigned char>(cloud->points[i].r * alpha + beta);
		cloud->points[i].g = cv::saturate_cast<unsigned char>(cloud->points[i].g * alpha + beta);
		cloud->points[i].b = cv::saturate_cast<unsigned char>(cloud->points[i].b * alpha + beta);
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Finished adjusting contrast." << endl;
	cout << "--------------------(" << duration.count() / 1000 << "ms)" << endl;
}

// Filters outliers(noises) from the point cloud.
void FilterNoise(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud, int meanK, float stdDevMulThresh) {
	auto start = high_resolution_clock::now();
	cout << "Start filtering noise." << endl;
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;

	sor.setInputCloud(cloud);
	sor.setMeanK(meanK);
	sor.setStddevMulThresh(stdDevMulThresh);
	sor.filter(*filteredCloud);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Finished filtering noise" << endl;
	cout << "--------------------(" << duration.count() / 1000 << "ms)" << endl;
}

// Cluster the point cloud to players.
void ClusterObjects(vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float threshold, int minClusterSize, int maxClusterSize) {
	auto start = high_resolution_clock::now();
	cout << "Start clustering players." << endl;

	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(cloud);

    vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(threshold);
    ec.setMinClusterSize(minClusterSize);
    ec.setMaxClusterSize(maxClusterSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

	cout << "Number of clusters: " << cluster_indices.size() << endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);

	// Store the clustered points to a vector of point clouds
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++) {
			pcl::PointXYZRGB p = cloud->points[*pit];
			cloud_cluster->points.push_back(p);
		}
			
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::cout << "Cluster " << it - cluster_indices.begin()  << " contains " << cloud_cluster->points.size() << " data points." << std::endl;

		clusters.push_back(cloud_cluster);
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Finished clustering players." << endl;
	cout << "--------------------(" << duration.count() / 1000 << "ms)" << endl;
}

// Initialize Players with clustered point clouds and find the dominant colors.
void SetPlayers(vector<Player>& players, vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clusters, int dominantCount, int dominantClusterCount) {
	auto start = high_resolution_clock::now();
	//cout << "Start setting players and finding dominant colors." << endl;
	
	for (auto cluster : clusters) {
		Player p(cluster, dominantCount, dominantClusterCount);
		players.push_back(p);
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	//cout << "Finished setting players and finding dominant colors." << endl;
	//cout << "--------------------(" << duration.count() / 1000 << "ms)" << endl;
}

// **DEPRECATED** Use dominant colors of players to cluster players into 3 teams.
void ClusterTeams(vector<Player>& players) {
	auto start = high_resolution_clock::now();
	//cout << "Start clustering teams." << endl;

	int featureNum = players[0].GetDominantColors().rows;
	cv::Mat features(0, featureNum, CV_32F);
	for (auto player : players) {
		cv::Mat feature(1, 0, CV_32F);
		cv::Mat temp;
		player.GetDominantColors().convertTo(temp, CV_32F);
		feature = temp.reshape(0, 1);
		features.push_back(feature);
	}

	int teamCount = 3;
	cv::Mat labels, centers;
	cv::TermCriteria criteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 0.01);
	cv::kmeans(features, teamCount, labels, criteria, 3, cv::KMEANS_RANDOM_CENTERS, centers);

	for (int i = 0; i < players.size(); i++) {
		players[i].SetTeam(labels.at<int>(i, 0));
	}

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	//cout << "Finished clustering teams." << endl;
	//cout << "--------------------(" << duration.count() / 1000 << "ms)" << endl;
	return;
}

// Helper function colors player point clouds by their dominant colors or teams.
void ColorPlayers(vector<Player>& players, bool showDominantColors) {
	vector<vector<int>> colors = { { 255, 0, 0 },
								   { 0, 255, 0 },
								   { 0, 0, 255 } };
	for (Player player : players) {
		cv::Mat color = player.GetDominantColors();
		for (auto p = player.GetCloud()->begin(); p != player.GetCloud()->end(); p++) {
			if (showDominantColors)
			{
				if (color.rows > 1) {
					if (p->y > player.GetCenter()[1]) {
						p->r = color.at<float>(0, 2);
						p->g = color.at<float>(0, 1);
						p->b = color.at<float>(0, 0);
					}
					else {
						p->r = color.at<float>(1, 2);
						p->g = color.at<float>(1, 1);
						p->b = color.at<float>(1, 0);
					}
				}
				else {
					p->r = color.at<float>(0, 2);
					p->g = color.at<float>(0, 1);
					p->b = color.at<float>(0, 0);
				}
			}
			else {
				p->r = colors[player.GetTeam()][0];
				p->g = colors[player.GetTeam()][1];
				p->b = colors[player.GetTeam()][2];
			}
		}
		
	}
}

// Helper function calculates if number of the players in the teams are [3, 5, 5] (Ref, TeamA, TeamB);
bool CountTeams(vector<Player> players) {
	vector<int> teams = { 0, 0, 0 };
	for (auto p : players) {
		teams[p.GetTeam()]++;
	}
	sort(teams.begin(), teams.end());
	if (teams[0] == 3 && teams[1] == 5 && teams[2] == 5) {
		return true;
	}
	return false;
}

// Helper function outputs postions of each player in each team as result.
void OutputPosition(vector<Player> players) {
	vector<vector<float>> teamA;
	vector<vector<float>> teamB;
	vector<vector<float>> ref;
	vector<int> teamCount(3, 0);
	for (Player p : players) {
		teamCount[p.GetTeam()]++;
	}
	int teamAIdx = -1;
	int teamBIdx = -1;
	int refIdx = min_element(teamCount.begin(), teamCount.end()) - teamCount.begin();
	for (Player p : players) {
		if (p.GetTeam() == refIdx) {
			ref.push_back(vector<float>{p.GetCenter()[0], p.GetCenter()[2]});
		}
		else if (teamAIdx == -1 || p.GetTeam() == teamAIdx) {
			teamAIdx = p.GetTeam();
			teamA.push_back(vector<float>{p.GetCenter()[0], p.GetCenter()[2]});
		}
		else {
			teamBIdx = p.GetTeam();
			teamB.push_back(vector<float>{p.GetCenter()[0], p.GetCenter()[2]});
		}
	}
	cout << "TeamA: [";
	for (int i = 0; i < teamA.size(); i++) {
		cout << "[" << teamA[i][0] << ", " << teamA[i][1] << "]";
		if (i != teamA.size() - 1) cout << ",";
	}
	cout << "]\n";
	cout << "TeamB: [";
	for (int i = 0; i < teamB.size(); i++) {
		cout << "[" << teamB[i][0] << ", " << teamB[i][1] << "]";
		if (i != teamB.size() - 1) cout << ",";
	}
	cout << "]\n";
	cout << "Referee: [";
	for (int i = 0; i < ref.size(); i++) {
		cout << "[" << ref[i][0] << ", " << ref[i][1] << "]";
		if (i != ref.size() - 1) cout << ",";
	}
	cout << "]\n";
}

int main() {
	// Hyperparameters
	
	// Preprocessing parameters
	float brightnessAdjConstant = 1.4; // R, G, B values of each point are multiplied by this number.
	int noiseMeanK = 50; // Number of adjacent points considered while finding outliers. See https://pcl.readthedocs.io/projects/tutorials/en/latest/statistical_outlier.html.
	float noiseStdDevMulThresh = 0.5; // Distance larger than (noiseStdDevMulThresh * standard deviation) of the mean distance to the query point will be marked as outliers and removed.

	// Player clustering parameters
	float minObjectDistance = 0.8; // Minimum distance between clusters.
	int minClusterSize = 750; // Minimum points per cluster.
	int maxClusterSize = 5000; // Maximum points per cluster.

	// Color clustering parameters
	int dominantCount = 3; // Number of the most dominant colors extracted from a player.
	int dominantClusterCount = 3;

	// Player teaming parameters
	double featureWeight = 1.4; // See See "Cluster teams using players’ dominant colors" and "Result/Accuracy Optimization" sections in readme.
	int maxKmeansIteration = 10; // Maximum iterations for k-means clustering.
	
	srand(time(NULL));
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	if (!LoadCloud("point_cloud_data.txt", cloud)) return 0;

	BrightnessCorrection(cloud, brightnessAdjConstant, 10);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	FilterNoise(cloud, filteredCloud, noiseMeanK, noiseStdDevMulThresh);

	vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters;
	ClusterObjects(clusters, filteredCloud, minObjectDistance, minClusterSize, maxClusterSize);

	/*
	vector<Player> players = {};
	SetPlayers(players, clusters, dominantCount, dominantClusterCount);
	*/
	double a = 0;
	for (int i = 0; i < 10000; i++) {
	vector<Player> players = {};
	SetPlayers(players, clusters, dominantCount, dominantClusterCount);
	ClusterTeams(players); // DEPRECATED
	if (CountTeams(players)) a += 1.0/10000;
	}
	cout << a << endl;
	//Kmeans_PointBGR kmeans(3, maxKmeansIteration, featureWeight, players);

	//OutputPosition(players);
	
	/*ColorPlayers(players, false);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr mCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr c : clusters) {
		*mCloud += *c;
	}
	ViewPC(mCloud);*/
	
	return 0;
}