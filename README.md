
# Position Basketball Players
- Identify people from a point cloud.
- Team the players, including referees.
- Output position of players in each team.

# Environment
Any dependencie older than below is not tested.
- ### opencv 4.5.1
```bash
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
unzip opencv.zip
cd opencv-master && mkdir -p build && cd build
cmake  ..
make -j6
sudo make -j6 install
```
- ### vtk 8.2
```bash
sudo apt-get install libxt-dev libgl1-mesa-dev
wget https://www.vtk.org/files/release/8.2/VTK-8.2.0.tar.gz
tar xvzf VTK-8.2.0.tar.gz
cd VTK-8.2.0 && mkdir build && cd build
ccmake ..
```
press c 
change CMAKE_BUILD_TYPE to Release
press c 
press g

```bash
make -j6
sudo make -j6 install
```
- ### pcl 1.11.0
```bash
sudo apt install libboost-all-dev libeigen3-dev libflann-dev libpcl-dev
```

# Installation
```bash
cd SecondSpectrum
mkdir build && cp ./point_cloud_data.txt ./build && cd build
cmake ..
make -j6
./SecondSpectrum
```

# Data Preparation
Store point cloud information in "point_cloud_data.txt".
Each line of the input file has 6 columns (X, Y, Z, R, G, B)
X, Y, Z: the xyz position of the 3d point
R, G, B: the rgb component of the colour of the 3d point
For example:
21.3, 1.1, -3.4, 25, 14, 10
-1.3 2.7 14.4 255 11 5
The above file has 2 3d points:
>Position: (21.3, 1.1, -3.4), Colour: (25, 14, 10)
>Position: (-1.3, 2.7, 14.4), Colour: (255, 11, 5)

# Execute the Program
Windows 10: Put the point_cloud_data.txt in Release folder and run the SecondSpectrum.exe.

# Architecture
![](https://imgur.com/Kx01xg8.png)


# Hyperparameters
### Preprocessing
- int `noiseMeanK = 50`: Number of adjacent points considered while finding outliers.
See https://pcl.readthedocs.io/projects/tutorials/en/latest/statistical_outlier.html.
- float `noisestdDevMulThresh = 0.5`: Distance larger than (`noiseStdDevMulThresh` * standard deviation) of the mean distance to the query point will be marked as outliers and removed.

### Point Cloud Clustering
- float `minObjectDistance = 0.8`: Minimum distance between clusters.
- int `minClusterSize = 750`: Minimum points per cluster.
- int `maxClusterSize = 5000`: Maximum points per cluster.

### Team Clustering
- int `dominantCount = 3`: Number of the most dominant colors extracted from a player.
- int `dominantClusterCount = 3`: Number of the clusters that each player's colors are clustered to.

# Stages
### Load txt files to pcl (~25 ms)
Function `LoadCloud` loads points from txt to a pcl XYZRGB point and push it to a pcl point cloud.
![](https://imgur.com/dLH6lT7.png)

### Filter noises (~78 ms)
Function `FilterNoise` uses pcl built-in function `StatisticalOutlierRemoval` to filter outliers(noises) from the point cloud.
>Reference:: https://pcl.readthedocs.io/projects/tutorials/en/latest/statistical_outlier.html.

![](https://imgur.com/GwfVQyw.png)
### Cluster players (~276 ms)
Function `ClusterObjects` uses pcl built-in function `EuclideanClusterExtraction` to cluster the point cloud to players.
This function excludes clusters with points fewer than `minClusterSize`.
>Reference: https://pcl.readthedocs.io/en/latest/cluster_extraction.html

Color each player with a random color:![](https://imgur.com/U8GuQaf.png)
### Find dominant colors of each player (~17 ms)
![](https://imgur.com/7VCtuxa.png)`Class Player`'s constructor takes a clustered point cloud as input.
Function `Player::FindCenter` loops through the input point cloud and finds the average coordination to be the player center.
Function `Player::Kmeans` clusters colors of points in the input point cloud into `dominantClusterCount`  of clusters using `kmeans` function from Opencv.
>Reference: https://docs.opencv.org/4.5.1/d5/d38/group__core__cluster.html

Extract the centers of the color clusters that have the most points to be each player's feature.
Function `Player::SortColor` sorts the extracted dominant colors by the length of the RGB vector.

Demostrating the top two dominant colors for each player: 
![](https://imgur.com/wRBcOx1.png)
The color at upper body is the first dominant color, and the color at lower body is the second dominant color.
### Cluster teams using players' dominant colors (~0 ms)

 Function `ClusterTeams` clusters the players by players' dominant colors.
- Horizontally concatenate dominant colors of each player to a 1 x (`dominantCount` x 3) feature.
For example: [b1, g1, r1, b2, g2, r2, b3, g3, r3] for a player with 3 dominant colors
- Utilize opencv kmeans function to cluster players into 3 teams.

Color players by their teams:
![](https://imgur.com/ta89czJ.png)

### Output player positions

Function `OutputPosition` extracts players' team labels, sets the team with the fewest player to be the referees, and output the X-Z coordinate of each player's center in each team.

![](https://imgur.com/tQGxYOc.png)

# Result
Overall, the pipeline successfully outputs positions for all the layers belonging to all the classes.

## Robustness

The pipeline achieves 97% of accuracy in making team predictions to the sample input. This result is based on running the pipeline 10,000 times.

Due to the <u>randomness and uncertainty of k-means clustering</u>, there is no guarantee that the output results are identical even with the same input and hyperparameters.

On the other hand, the robustness of the pipeline and this set of hyperparameters still need to be examined  by more data.

## Efficiency

Testing environment: OS: Windows 10, CPU: AMD 5600X

The pipeline takes about 396ms (~2.5 fps) from loading the txt file to output the result. 
It is still far from being real-time. The bottleneck is the point cloud clustering stage.

# TODO: Possible Future Improvement

- **Robustness** - Obtain more data with team labels -> Train multi-class SVM or other classification models.
- **Efficiency**
	1. Downsample the point cloud points to voxels before clustering.
	2. Obtain more data with team labels -> Train a 3d point cloud segmentation model, e.g., PointCNN, and find the center of points within the segments. This method does not need the time-consuming point cloud clustering step.

# Further Discussion
This pipeline has difficulty in following situations:
- **Clustering crowded players**: Using 3D point cloud segmentation models could be a solution.
- **Team A and B's jerseys have similar colors**: Need to use CNN models to extract point cloud features or label the point cloud in advance using object detection and classification models based on 2D images.
