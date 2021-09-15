#pragma once
#include <vector>
#include <opencv2/opencv.hpp> 
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/pcl_search.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/point_types_conversion.h>


class Player {
public:
	Player(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int dominantCount, int dominantClusterCount);
	void SetTeam(int t);
	int GetTeam();
	cv::Mat GetDominantColors();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr GetCloud();
	std::vector<float> GetCenter();
private:
	int team = 0;
	int dominantClusterCount = 0;
	std::vector<float> center = {0, 0, 0};
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
	cv::Mat dominantColors;
	cv::Mat SortColor(cv::Mat m);
	cv::Mat Kmeans(cv::Mat points, int clusterCount);
	void FindDominantColors(int dominantCount);
	void FindCenter();
};