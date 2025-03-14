#include <string>
#include <thread>

#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>
#include <vector>
#include <pcl/registration/ndt.h>

#include <Eigen/src/Core/Matrix.h>

#include "livox_ros_driver/CustomMsg.h"
#include "fast_gicp/gicp/fast_gicp.hpp"
#include "ros/console.h"

constexpr float min_thr       = 1.5f;
constexpr float local_map_thr = 25.f;

bool inited              = false;
Eigen::Matrix4f veh_pose = Eigen::Matrix4f::Identity();

pcl::PointCloud<pcl::PointXYZ>::Ptr points_map(new pcl::PointCloud<pcl::PointXYZ>);
sensor_msgs::PointCloud2 cloud_msg;
nav_msgs::Path path_msg;

ros::Publisher points_pub;
ros::Publisher pose_pub;
ros::Publisher path_pub;

std::ofstream tum_stream;

ros::Time t_msg;

void pub_path(Eigen::Matrix4f &pose, bool write_file = true) {
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp    = t_msg;
    pose_msg.header.frame_id = "map";
    pose_msg.pose.position.x = pose(0, 3);
    pose_msg.pose.position.y = pose(1, 3);
    pose_msg.pose.position.z = pose(2, 3);
    Eigen::Quaternionf quat(pose.block<3, 3>(0, 0));
    pose_msg.pose.orientation.x = quat.x();
    pose_msg.pose.orientation.y = quat.y();
    pose_msg.pose.orientation.z = quat.z();
    pose_msg.pose.orientation.w = quat.w();
    pose_pub.publish(pose_msg);

    if (write_file) {
        // 获取当前时间戳并转换为 TUM 格式所需的时间戳格式
        std::stringstream timestamp;
        timestamp << std::fixed << std::setprecision(6) << pose_msg.header.stamp.toSec();

        // 将 Odometry 消息的位姿信息保存为 TUM 格式的一行
        std::string line = timestamp.str() + " " +                             //
                           std::to_string(pose_msg.pose.position.x) + " " +    //
                           std::to_string(pose_msg.pose.position.y) + " " +    //
                           std::to_string(pose_msg.pose.position.z) + " " +    //
                           std::to_string(pose_msg.pose.orientation.x) + " " + //
                           std::to_string(pose_msg.pose.orientation.y) + " " + //
                           std::to_string(pose_msg.pose.orientation.z) + " " + //
                           std::to_string(pose_msg.pose.orientation.w);

        // 将行写入文件
        tum_stream << line << std::endl;
    }

    // 发布机器人轨迹
    path_msg.header.stamp    = t_msg;
    path_msg.header.frame_id = "map";
    path_msg.poses.push_back(pose_msg);
    path_pub.publish(path_msg);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr get_local_map(
    pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_map, pcl::PointXYZ &center_point, float thr) {
    // 创建一个新的点云，用于存储剪裁后的点
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_map(new pcl::PointCloud<pcl::PointXYZ>);

    // 遍历所有点，将距离小于20米的点添加到剪裁后的点云中
    for (const auto &point : *pcl_map) {
        double distance = pcl::euclideanDistance(center_point, point);
        if (distance <= thr) {
            local_map->push_back(point);
        }
    }
    return local_map;
};

void registration(pcl::PointCloud<pcl::PointXYZ>::Ptr &points, pcl::PointCloud<pcl::PointXYZ>::Ptr &local_map) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
    if (!inited) {
        inited = true;

        pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

        // 设置要匹配的目标点云
        ndt.setInputSource(points);
        ndt.setInputTarget(local_map);

        ndt.setTransformationEpsilon(1e-8);
        ndt.setStepSize(0.1);
        ndt.setMaximumIterations(100);
        ndt.setResolution(1.0);

        ndt.align(*aligned, veh_pose);

        veh_pose = ndt.getFinalTransformation();
        ROS_INFO(
            "veh pose init finish: iter: %d x: %f y: %f!", ndt.getFinalNumIteration(), veh_pose(0, 3), veh_pose(1, 3));

    } else {
        fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
        gicp.setInputSource(points);
        gicp.setInputTarget(local_map);
        gicp.setNumThreads(8);
        gicp.setMaxCorrespondenceDistance(0.5);
        gicp.align(*aligned, veh_pose);
        Eigen::Matrix4f gicp_tf_matrix;
        veh_pose = gicp.getFinalTransformation();
    }

    sensor_msgs::PointCloud2 points_msg;
    pcl::toROSMsg(*aligned, points_msg);
    points_msg.header.stamp    = t_msg;
    points_msg.header.frame_id = "map";

    points_pub.publish(points_msg);

    pub_path(veh_pose);
}

void livox_callback(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::PointCloud<pcl::PointXYZ>::Ptr points(new pcl::PointCloud<pcl::PointXYZ>);
    for (uint i = 1; i < msg->point_num; i++) {
        if (msg->points.at(i).x * msg->points.at(i).x + msg->points.at(i).y * msg->points.at(i).y +
                msg->points.at(i).z * msg->points.at(i).z <
            min_thr * min_thr) {
            continue;
        }
        pcl::PointXYZ pt;
        pt.x = msg->points.at(i).x;
        pt.y = msg->points.at(i).y;
        pt.z = msg->points.at(i).z;
        points->push_back(pt);
    }

    t_msg = msg->header.stamp;

    pcl::PointCloud<pcl::PointXYZ>::Ptr local_map(new pcl::PointCloud<pcl::PointXYZ>);
    auto center = pcl::PointXYZ(veh_pose(0, 3), veh_pose(1, 3), veh_pose(2, 3));
    local_map   = get_local_map(points_map, center, local_map_thr);

    registration(points, local_map);

    auto end                              = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    ROS_INFO("Localization Time: %.2f[ms]", elapsed.count() * 1e3);
}

void pcl_callback(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::PointCloud<pcl::PointXYZ>::Ptr points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *points);

    pcl::PointCloud<pcl::PointXYZ>::Ptr points_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    for (auto &pt : *points) {
        if (pt.x * pt.x + pt.y * pt.y + pt.z * pt.z < min_thr * min_thr) {
            continue;
        }
        points_filtered->push_back(pt);
    }

    t_msg = msg->header.stamp;

    pcl::PointCloud<pcl::PointXYZ>::Ptr local_map(new pcl::PointCloud<pcl::PointXYZ>);
    auto center = pcl::PointXYZ(veh_pose(0, 3), veh_pose(1, 3), veh_pose(2, 3));
    local_map   = get_local_map(points_map, center, local_map_thr);

    registration(points_filtered, local_map);

    auto end                              = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    ROS_INFO("Localization Time: %.2f[ms]", elapsed.count() * 1e3);
}

void pointCloudPublisherThread(ros::Publisher &pub) {
    // 发布点云消息
    ros::Rate loop_rate(1);
    while (ros::ok()) {
        pub.publish(cloud_msg);

        // publish guess pose
        pub_path(veh_pose, false);

        loop_rate.sleep();
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "locate_node");
    ros::NodeHandle nh("~");

    int points_type;
    std::string map_file_path, topic_name;
    nh.param<std::string>(
        "map_file_path", map_file_path, ros::package::getPath("fast_lio") + "/PCD/points_map_20240310.pcd");
    nh.param<int>("points_type", points_type, 0); // 0: livox, 1:pointcloud2
    nh.param<std::string>("topic_name", topic_name, "/livox/lidar");

    std::vector<float> guess_pose;
    if (nh.getParam("guess_pose", guess_pose)) { // [x y z roll pitch yaw]
        veh_pose.topLeftCorner(3, 3) = (Eigen::AngleAxisf(guess_pose.at(5), Eigen::Vector3f::UnitZ()) *
                                        Eigen::AngleAxisf(guess_pose.at(4), Eigen::Vector3f::UnitY()) *
                                        Eigen::AngleAxisf(guess_pose.at(3), Eigen::Vector3f::UnitX()))
                                           .toRotationMatrix();

        veh_pose(0, 3) = guess_pose.at(0);
        veh_pose(1, 3) = guess_pose.at(1);
        veh_pose(2, 3) = guess_pose.at(2);
    } else {
        ROS_ERROR("Parameter guess_pose failed to get value");
    }

    float map_yaw, map_roll, map_pitch;
    nh.param<float>("map_roll", map_roll, 0.0f);
    nh.param<float>("map_pitch", map_pitch, 0.0f);
    nh.param<float>("map_yaw", map_yaw, 0.0f);

    std::string tum_save_path;
    nh.param<std::string>("tum_save_path", tum_save_path, "src/mid360_locate/tum/localization_result.txt");
    tum_stream.open(tum_save_path);
    if (!tum_stream.is_open()) {
        ROS_ERROR("Failed to open file: %s", tum_save_path.c_str());
        return -1;
    }

    ROS_INFO("Reading map file: %s.", map_file_path.c_str());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_file_path, *points_map) == -1) {
        PCL_ERROR("Couldn't read file.\n");
        return (-1);
    }

    ROS_INFO("Original point cloud size: %zu points.", points_map->size());
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(points_map);
    sor.setLeafSize(0.1f, 0.1f, 0.1f); // 设置体素大小
    sor.filter(*points_map);           // 执行滤波
    ROS_INFO("Filtered point cloud size: %zu points.", points_map->size());

    ROS_INFO("Find the plane...");
    pcl::ModelCoefficients coefficients;
    pcl::PointIndices inliers;
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.01);
    seg.setAxis(Eigen::Vector3f(0, 0, 1));
    seg.setInputCloud(points_map);
    seg.segment(inliers, coefficients);

    ROS_INFO(
        "Plane coefficients: %f %f %f %f",
        coefficients.values[0],
        coefficients.values[1],
        coefficients.values[2],
        coefficients.values[3]);

    ROS_INFO("Rotate map to z plane.");
    // 已知平面的法线和水平面的法线
    Eigen::Vector3f plane_normal(
        coefficients.values[0], coefficients.values[1], coefficients.values[2]); // 假设平面的法线向量
    plane_normal.normalize();                                                    // 需要对法线向量进行归一化
    Eigen::Vector3f horizontal_normal(0.0, 0.0, 1.0); // 假设水平面的法线向量为(0, 0, 1)
    // 计算旋转矩阵
    Eigen::Quaternionf rotation     = Eigen::Quaternionf::FromTwoVectors(plane_normal, horizontal_normal);
    Eigen::Matrix3f rotation_matrix = rotation.toRotationMatrix();

    Eigen::Matrix4f T_to_plane       = Eigen::Matrix4f::Identity();
    T_to_plane(2, 3)                 = coefficients.values[3];
    T_to_plane.topLeftCorner<3, 3>() = rotation_matrix;
    pcl::transformPointCloud(*points_map, *points_map, T_to_plane);

    ROS_INFO("Rotate map's yaw.");

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(
        Eigen::AngleAxisf(map_yaw, Eigen::Vector3f::UnitZ()) *   //
        Eigen::AngleAxisf(map_pitch, Eigen::Vector3f::UnitY()) * //
        Eigen::AngleAxisf(map_roll, Eigen::Vector3f::UnitX()));

    // 应用旋转变换到点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*points_map, *points_map, transform);

    ROS_INFO("Map init finsh.");

    // 将pcl点云转换为ROS消息
    pcl::toROSMsg(*points_map, cloud_msg);
    // 设置消息的元数据
    cloud_msg.header.stamp    = t_msg;
    cloud_msg.header.frame_id = "map";

    ros::Subscriber points_sub;
    if (points_type == 0) {
        points_sub = nh.subscribe(topic_name, 1, livox_callback);
        ROS_INFO("Subscribe livox pointcloud");
    } else {
        points_sub = nh.subscribe(topic_name, 1, pcl_callback);
        ROS_INFO("Subscribe ros pointcloud2");
    }
    points_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned", 1);
    pose_pub   = nh.advertise<geometry_msgs::PoseStamped>("/veh_pose", 10);
    path_pub   = nh.advertise<nav_msgs::Path>("/path", 10);

    ros::Publisher map_pub = nh.advertise<sensor_msgs::PointCloud2>("/points_map", 1);
    std::thread publisher_thread(pointCloudPublisherThread, std::ref(map_pub));

    ros::spin();

    tum_stream.close();

    return 0;
}
