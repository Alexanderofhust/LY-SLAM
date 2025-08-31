#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "MPCSolver.hpp"
#include "SystemModel.hpp"
#include "Types.hpp"

namespace mpc_ros {

    class MPCNode : public rclcpp::Node {
    public:
        explicit MPCNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
        virtual ~MPCNode();

    private:
        // ROS2 subscribers
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
        rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;

        // ROS2 publishers
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr predicted_path_pub_;

        // TF2
        std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

        // MPC components
        mpc::MPCSolver solver_;
        mpc::SystemModel model_;

        // State and reference
        mpc::State current_state_;
        mpc::ReferenceTrajectory reference_trajectory_;
        bool has_odom_{false};
        bool has_reference_{false};

        // Parameters
        double control_rate_{10.0};
        std::string robot_frame_{"base_link"};
        std::string world_frame_{"odom"};

        // Timer for control loop
        rclcpp::TimerBase::SharedPtr control_timer_;

        // Callbacks
        void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
        void pathCallback(const nav_msgs::msg::Path::SharedPtr msg);
        void controlTimerCallback();

        // Helper methods
        bool getRobotPose(geometry_msgs::msg::PoseStamped& robot_pose);
        void publishCommand(const mpc::ControlInput& control_input);
        void publishPredictedPath(const std::vector<mpc::State>& predicted_states);

        // Parameter handling
        void declareParameters();
        void loadParameters();
    };

}  // namespace mpc_ros