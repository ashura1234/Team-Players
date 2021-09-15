#include "Player.h"

Player::Player(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int dominantCount, int dominantClusterCount) {
	this->cloud = cloud;
	this->dominantClusterCount = dominantClusterCount;
	FindCenter();
	FindDominantColors(dominantCount);
}

void Player::SetTeam(int t){
	this->team = t;
}

int Player::GetTeam() {
	return this->team;
}

cv::Mat Player::GetDominantColors() {
	return this->dominantColors;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Player::GetCloud() {
	return this->cloud;
}

std::vector<float> Player::GetCenter() {
	return this->center;
}

cv::Mat Player::SortColor(cv::Mat m) {
	cv::Mat res(m.rows, m.cols, m.type());
	cv::Mat sortIndices;
	cv::Mat length(m.rows, 1, CV_32F);
	for (int i = 0; i < m.rows; i++) {
		length.at<float>(i, 0) = m.row(i).dot(m.row(i));
	}
	cv::sortIdx(length, sortIndices, cv::SORT_EVERY_COLUMN | cv::SORT_DESCENDING);
	for (int i = 0; i < sortIndices.rows; i++) {
		m.row(sortIndices.at<int>(i, 0)).copyTo(res.row(i));
	}
	return res;
}

cv::Mat Player::Kmeans(cv::Mat src, int clusterCount) {
	cv::Mat points;
	src.convertTo(points, CV_32F);
	points = points.reshape(1, points.total());
	cv::Mat labels, centers;
	cv::TermCriteria criteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 0.1);
	cv::kmeans(points, clusterCount, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);
	

	centers = centers.reshape(3, centers.rows);
	points = points.reshape(3, points.rows);

	cv::Mat colorCount(clusterCount, 1, CV_32S, cv::Scalar(0));
	for (int i = 0; i < labels.rows; i++) {
		colorCount.at<int>(labels.at<int>(i, 0), 0)++;
	}

	
	cv::Mat sortIndices;
	cv::sortIdx(colorCount, sortIndices, cv::SORT_EVERY_COLUMN | cv::SORT_DESCENDING);

	cv::Mat res(clusterCount, points.cols, points.type());
	for (int i = 0; i < sortIndices.rows; i++) {
		centers.row(sortIndices.at<int>(i, 0)).copyTo(res.row(i));
	}
	return res;
}

void Player::FindDominantColors(int dominantCount) {
	int n = this->cloud->points.size();
	cv::Mat points(n, 1, CV_8UC3, cv::Vec3b(0,0,0));
	for (int i = 0; i < n; i++) {
		cv::Vec3b& pixel = points.at<cv::Vec3b>(i, 0);
		pixel[0] = this->cloud->points[i].b;
		pixel[1] = this->cloud->points[i].g;
		pixel[2] = this->cloud->points[i].r;
	}
	cv::Mat clusterCenters = Kmeans(points, this->dominantClusterCount);
	cv::Mat topCenters = clusterCenters.rowRange(0, dominantCount);
	cv::Mat sortedCenters = SortColor(topCenters);

	this->dominantColors = sortedCenters;
}

void Player::FindCenter() {
	int n = this->cloud->points.size();
	for (auto p : this->cloud->points) {
		this->center[0] += p.x / n;
		this->center[1] += p.y / n;
		this->center[2] += p.z / n;
	}
}