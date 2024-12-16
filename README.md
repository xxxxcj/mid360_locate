# MID360 locate

## information
1. This repository is a simple software to locate in parking using mid360.
2. First you need to build the pointcloud map using [fast_lio](https://github.com/hku-mars/FAST_LIO).
3. Because the parking lot is flat, this program will first fit the point cloud map to the flat surface and then convert the point cloud map to the flat surface.

## dependency
+ [fast_gicp](https://github.com/SMRT-AIST/fast_gicp)
+ [fast_lio](https://github.com/hku-mars/FAST_LIO)
+ [livox_ros_driver](https://github.com/Livox-SDK/livox_ros_driver)
+ [livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2)
+ other dependency needed by above.

## install
```shell
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/SMRT-AIST/fast_gicp
git clone --recursive https://github.com/hku-mars/FAST_LIO
git clone https://github.com/Livox-SDK/livox_ros_driver
git clone https://github.com/Livox-SDK/livox_ros_driver2
git clone https://github.com/xxxxcj/mid360_locate

cd ..
catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=True
```

## how to use
1. Use fast_lio to create a pointcloud map;
2. Modify `map_file_path`, `point_type` and `guess_pose` in `run.launch`:
   + map_file_path: the pcd map created by fast_lio
   + point_type: what type of pointcloud the mid360 use? pointcloud2 or livox?
   + guess_pose: the guess initial pose in map
```shel
source devel/setup.bash
roslaunch mid360_locate run.launch
```