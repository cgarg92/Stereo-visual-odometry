# Introduction

Localization is an essential feature for autonomous vehicles and therefore Visual Odometry has been a well investigated area in robotics vision. Visual Odometry helps augment the information where conventional sensors such as wheel odometer and inertial sensors such as gyroscopes and accelerometers fail to give correct information. Visual odometry estimates vehicle motion from a sequence of camera images from an onboard camera. It produces full 6-DOF (degrees of freedom) motion estimate, that is the translation along the axis and rotation around each of co-ordinate axis.

 Over the years, visual odometry has evolved from using stereo images to monocular imaging and now incorporating LiDAR laser information which has started to become mainstream in upcoming cars with self-driving capabilities. It is also a prerequisite for applications like obstacle detection, simultaneous localization and mapping (SLAM) and other tasks. Visual-SLAM (VSLAM) is a much more evolved variant of visual odometry which obtain global, consistent estimate of robot path. The path drift in VSLAM is reduced by identifying loop closures.
 
Some of the challenges encountered by visual odometry algorithms are:
1.	Varying lighting conditions
2.	In-sufficient scene overlap between consecutive frames
3.	Lack of texture to accurately estimate motion

## Types of Visual Odometry

* ### Monocular Visual Odometry
A single camera is used to capture motion. Usually a five-point relative pose  estimation method is used to estimate motion, motion computed is on a relative scale. Typically used in hybrid methods where other sensor data is also available.

* ### Stereo Visual Odometry
A calibrated stereo camera pair is used which helps compute the feature depth between images at various time points. Computed output is actual motion (on scale). If only faraway features are tracked then degenerates to monocular case.

# Algorithm Description
Our implementation is a variation of [1] by Andrew Howard. We have used KITTI visual odometry [2] dataset for experimentation. All the computation is done on grayscale images. The top level pipeline is shown in figure 1.

* ## Input Image sequence
Capture  stereo image pair at time T and T+1. The images are then processed to compensate for lens distortion. To simplify the task of disparity map computation stereo rectification is done so that epipolar lines become parallel to horizontal. In KITTI dataset the input images are already corrected for lens distortion and stereo rectified.

* ## Feature Detection
Features are generated on left camera image at time T using FAST (Features from Accelerated Segment Test) corner detector. FAST is computationally less expensive than other feature detectors like SIFT and SURF. To accurately compute the motion between image frames, feature bucketing is used. The image is divided into several non-overlapping rectangles and a maximum number (10) of feature points  with highest response value are then selected from each bucket. There are two benefits of bucketing: i) Input features are well distributed throughout the image which results in higher accuracy in motion estimation. ii) Due to less number of features computation complexity of algorithm is reduced which is a requirement in low-latency applications. Disparity map for time T is also generated using the left and right image pair.

* ## Feature Tracking
Features generated in previous step are then searched in image at time T+1. The original paper [1] does feature matching by computing the feature descriptors and then comparing them from images at both time instances. More recent literature uses KLT (Kanade-Lucas-Tomasi) tracker for feature matching. Features from image at time T are tracked at time T+1 using a 15x15 search windows and 3 image pyramid level search. KLT tracker outputs the corresponding coordinates for each input feature and accuracy and error measure by which each feature was tracked. Feature points that are tracked with high error or lower accuracy are dropped from further computation.

* ## 3D Point Cloud Generation
Now that we have the 2D points at time T and T+1, corresponding 3D points with respect to left camera are generated using disparity information and camera projection matrices. For each feature point a system of equations is formed for corresponding 3D coordinates (world coordinates) using left, right image pair and it is solved using singular value decomposition.

* ## Inlier Detection
Instead of an outlier rejection algorithm this algorithms uses an inlier detection algorithm which exploits the rigidity of scene points to find a subset of consistent 3D points at both time steps. The key idea here is the observation that although the absolute position of two feature points will be different at different time points the relative distance between them remains the same. If any such distance is not same, then either there is an error in 3D triangulation of at least one of the two features, or we have triangulated is moving, which we cannot use in the next step. 

* ## Motion Estimation
Frame to frame camera motion is estimated by minimizing the image re-projection error for all matching feature points. Image re-projection here means that for a pair of corresponding matching points Ja and Jb at time T and T+1, there exits corresponding world coordinates Wa and Wb. The world coordinates are re-projected back into image to estimate the 2D points coordinate for complementary time step and the distance between the true and projected 2D point is minimized using Levenberg-Marquardt least square optimization.  

## What is Visual Odometry?

Visual Odometry is the estimating the motion of a camera in real time using sequential images. The output that is obtained from a visual odometry algorithm are the 6 degrees of freedom of the moving object. The idea was first introduced for planetary rovers operating on Mars – Moravec 1980.

### VSLAM

Visual Odometry is used as a building block of a larger problem known as Simultaneous Localization and Mapping (SLAM). The goal of SLAM is to obtain a global, consistent estimate of the camera's path. SLAM finds loop closures in the paths generated by Visual Odometry and adjusts the drifts in the path.

### Applications of Visual Odometry

* The most important application of Visual Odometry is prediction of the trajectory of a moving robot/vehicle in uneven or slippery terrain. In uneven and slippery terains, wheels tend to slip and in such scenarios, wheel rotation calculations become unrealiable and visual odometry algorithms are used to give more accurate estimates of motion.

* Motion estimation for vehicles 
  * HD mapping
  * Autonomous cars

* Possible use in AR/VR applications

### Challenges 

* Robustness to lighting conditions
* Lack of features / non-overlapping images 
* Without loop closure the estimate still drifts

### Types of Visual Odometry

* Monocular Visual Odometry 
  * A single camera is used.
  * Estimates are relative.
* Stereo Visual Odometry
  * In Stereo Visual Odometry, a left and right camera are used.
  * Estimates are absolute.
* Augmented Stereo Visual Odometry
  * Visual Odometry can be augmented by using data from various sensors such as Lidar, Time of flight data, RGB-Depth and GPS.

## The Dataset

* The dataset we used was the KITTI Vision Benchmark Suite dataset by KIT, Germany. 
* The dataset included undistorted and stereo rectified grayscale and color image data along with LIDAR laser data.
* The calibration camera projection matrices are provided.
* The output poses for each frame are with respect to the 0th image in the sequence

## Our Approach

Our approach consists of the following steps.

### Feature Detection
 
We used the FAST (Features from Accelerated Segment Test) corner detection method for feature detection. 
* To ensure that the features we got from the FAST algorithm were spread out and not concentrated in a certain region, we used   feature bucketing in which we divide our image into a grid and only take a ------------------ number of features from each     part of the grid.

### Feature Matching
 
For feature matching between images at time t and t+1, we used Kanade–Lucas–Tomasi feature tracker with a search window size of 15x15 and 3 pyramid levels. To remove noise, we also did some pruning at this stage.

### 3D Point Triangulation

EDIT, ADD DETAILS
Corresponding feature points (u,v) from left and right image
Disparity displaced right image coordinate
Least square minimization on system of equations for (𝑋𝑤) ⃗, solved using SVD

### Inlier Detection

We used the Inlier Detection method to find the largest subset of consistent matches. We defined a match as a pair of points in which the distance between the points was the nearly the same in the images at time t and t+1. 

### Motion Estimate

We used the Levenberg-Marquardt least squares estimation to minimize the re-projection error which was expresses as e= ∑((𝑗𝑎 −𝑃Δ𝑤𝑏)^(2)+(𝑗𝑏 −𝑃Δ^(−1) 𝑤𝑎)^(2)) PUT PICTURE OF EQUATION HERE.

## Results

