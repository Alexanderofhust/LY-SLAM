#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "mpc_osqp_ros2/MPCNode.hpp"

class MPCNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        rclcpp::init(0, nullptr);

        // 创建节点选项并设置参数
        rclcpp::NodeOptions options;
        options.append_parameter_override("prediction_horizon", 10);
        options.append_parameter_override("control_horizon", 5);
        options.append_parameter_override("time_step", 0.1);

        // 创建节点
        node_ = std::make_shared<mpc_ros::MPCNode>(options);

        // 创建测试发布器和订阅器
        odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
        path_pub_ = node_->create_publisher<nav_msgs::msg::Path>("plan", 10);

        // 创建测试订阅器来接收控制命令
        cmd_vel_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
                "cmd_vel", 10,
                [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
                    last_cmd_vel_ = *msg;
                    cmd_vel_received_ = true;
                });

        // 创建测试订阅器来接收预测路径
        predicted_path_sub_ = node_->create_subscription<nav_msgs::msg::Path>(
                "predicted_path", 10,
                [this](const nav_msgs::msg::Path::SharedPtr msg) {
                    last_predicted_path_ = *msg;
                    predicted_path_received_ = true;
                });
    }

    void TearDown() override {
        node_.reset();
        rclcpp::shutdown();
    }

    void publishOdometry(double x, double y, double theta) {
        auto odom_msg = std::make_unique<nav_msgs::msg::Odometry>();
        odom_msg->header.stamp = node_->now();
        odom_msg->header.frame_id = "odom";
        odom_msg->child_frame_id = "base_link";

        odom_msg->pose.pose.position.x = x;
        odom_msg->pose.pose.position.y = y;
        odom_msg->pose.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, theta);
        odom_msg->pose.pose.orientation = tf2::toMsg(q);

        odom_pub_->publish(std::move(odom_msg));
    }

    void publishPath(const std::vector<std::tuple<double, double, double>>& waypoints) {
        auto path_msg = std::make_unique<nav_msgs::msg::Path>();
        path_msg->header.stamp = node_->now();
        path_msg->header.frame_id = "odom";

        for (const auto& wp : waypoints) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header.stamp = node_->now();
            pose.header.frame_id = "odom";

            pose.pose.position.x = std::get<0>(wp);
            pose.pose.position.y = std::get<1>(wp);
            pose.pose.position.z = 0.0;

            tf2::Quaternion q;
            q.setRPY(0, 0, std::get<2>(wp));
            pose.pose.orientation = tf2::toMsg(q);

            path_msg->poses.push_back(pose);
        }

        path_pub_->publish(std::move(path_msg));
    }

    void spinFor(double seconds) {
        rclcpp::Time start_time = node_->now();
        while (rclcpp::ok() && (node_->now() - start_time).seconds() < seconds) {
            rclcpp::spin_some(node_);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr predicted_path_sub_;

    geometry_msgs::msg::Twist last_cmd_vel_;
    nav_msgs::msg::Path last_predicted_path_;
    bool cmd_vel_received_ = false;
    bool predicted_path_received_ = false;
};

// 测试节点初始化
TEST_F(MPCNodeTest, NodeInitialization) {
EXPECT_NE(node_, nullptr);
EXPECT_TRUE(node_->get_parameter("prediction_horizon").as_int() == 10);
EXPECT_TRUE(node_->get_parameter("control_horizon").as_int() == 5);
EXPECT_TRUE(node_->get_parameter("time_step").as_double() == 0.1);
}

// 测试里程计回调
TEST_F(MPCNodeTest, OdometryCallback) {
publishOdometry(1.0, 2.0, 0.5);
spinFor(0.1);

// 节点应该成功接收并处理里程计消息
// 这里我们主要测试回调函数不会崩溃
SUCCEED();
}

// 测试路径回调
TEST_F(MPCNodeTest, PathCallback) {
std::vector<std::tuple<double, double, double>> waypoints = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {2.0, 0.0, 0.0}
};

publishPath(waypoints);
spinFor(0.1);

// 节点应该成功接收并处理路径消息
SUCCEED();
}

// 测试控制循环
TEST_F(MPCNodeTest, ControlLoop) {
// 发布初始位姿和路径
publishOdometry(0.0, 0.0, 0.0);

std::vector<std::tuple<double, double, double>> waypoints;
for (int i = 0; i < 10; ++i) {
waypoints.emplace_back(i * 0.5, 0.0, 0.0);
}
publishPath(waypoints);

// 等待控制循环执行
spinFor(0.5);

// 验证是否收到控制命令
EXPECT_TRUE(cmd_vel_received_);

// 验证控制命令的合理性
EXPECT_GE(last_cmd_vel_.linear.x, -0.5); // 线速度不低于最小约束
EXPECT_LE(last_cmd_vel_.linear.x, 0.5);  // 线速度不超过最大约束
EXPECT_GE(last_cmd_vel_.angular.z, -1.0); // 角速度不低于最小约束
EXPECT_LE(last_cmd_vel_.angular.z, 1.0);  // 角速度不超过最大约束

// 验证是否收到预测路径
EXPECT_TRUE(predicted_path_received_);
EXPECT_GT(last_predicted_path_.poses.size(), 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}