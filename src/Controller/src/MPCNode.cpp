#include "mpc_osqp_ros2/MPCNode.hpp"
#include <chrono>
#include <memory>
#include <string>

using namespace std::chrono_literals;

namespace mpc_ros {

    MPCNode::MPCNode(const rclcpp::NodeOptions & options)
            : Node("mpc_controller", options) {
        // Initialize TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Declare and load parameters
        declareParameters();
        loadParameters();

        // Initialize MPC solver
        mpc::MPCConfig config;
        config.prediction_horizon = this->get_parameter("prediction_horizon").as_int();
        config.control_horizon = this->get_parameter("control_horizon").as_int();
        config.time_step = this->get_parameter("time_step").as_double();

        // Load weights from parameters
        auto state_weights = this->get_parameter("state_weights").as_double_array();
        auto input_weights = this->get_parameter("input_weights").as_double_array();
        auto input_rate_weights = this->get_parameter("input_rate_weights").as_double_array();

        config.state_weight = Eigen::Map<Eigen::Matrix3d>(state_weights.data());
        config.input_weight = Eigen::Map<Eigen::Matrix2d>(input_weights.data());
        config.input_rate_weight = Eigen::Map<Eigen::Matrix2d>(input_rate_weights.data());

        solver_.setup(config);

        // Initialize system model
        mpc::ModelParameters model_params;
        model_params.wheel_radius = this->get_parameter("wheel_radius").as_double();
        model_params.wheel_base = this->get_parameter("wheel_base").as_double();
        model_params.mass = this->get_parameter("mass").as_double();
        model_params.moment_of_inertia = this->get_parameter("moment_of_inertia").as_double();

        model_.setParameters(model_params);

        // Set up constraints
        mpc::Constraints constraints;
        auto min_input = this->get_parameter("min_input").as_double_array();
        auto max_input = this->get_parameter("max_input").as_double_array();
        auto min_state = this->get_parameter("min_state").as_double_array();
        auto max_state = this->get_parameter("max_state").as_double_array();

        constraints.setInputConstraints(
                mpc::ControlInput(min_input[0], min_input[1]),
                mpc::ControlInput(max_input[0], max_input[1])
        );

        constraints.setStateConstraints(
                mpc::State(min_state[0], min_state[1], min_state[2]),
                mpc::State(max_state[0], max_state[1], max_state[2])
        );

        // Create subscribers
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
                "odom", 10, std::bind(&MPCNode::odomCallback, this, std::placeholders::_1));

        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
                "plan", 10, std::bind(&MPCNode::pathCallback, this, std::placeholders::_1));

        // Create publishers
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
        predicted_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("predicted_path", 10);

        // Create control timer
        control_timer_ = this->create_wall_timer(
                std::chrono::duration<double>(1.0 / control_rate_),
                std::bind(&MPCNode::controlTimerCallback, this));

        RCLCPP_INFO(this->get_logger(), "MPC Controller node initialized");
    }

    MPCNode::~MPCNode() {
        // Cleanup if needed
    }

    void MPCNode::declareParameters() {
        // MPC parameters
        this->declare_parameter("prediction_horizon", 20);
        this->declare_parameter("control_horizon", 10);
        this->declare_parameter("time_step", 0.1);

        // Weight parameters
        std::vector<double> state_weights = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 5.0};
        std::vector<double> input_weights = {1.0, 0.0, 0.0, 1.0};
        std::vector<double> input_rate_weights = {0.1, 0.0, 0.0, 0.1};

        this->declare_parameter("state_weights", state_weights);
        this->declare_parameter("input_weights", input_weights);
        this->declare_parameter("input_rate_weights", input_rate_weights);

        // Model parameters
        this->declare_parameter("wheel_radius", 0.1);
        this->declare_parameter("wheel_base", 0.5);
        this->declare_parameter("mass", 10.0);
        this->declare_parameter("moment_of_inertia", 2.0);

        // Constraint parameters
        std::vector<double> min_input = {-0.5, -1.0};
        std::vector<double> max_input = {0.5, 1.0};
        std::vector<double> min_state = {-10.0, -10.0, -3.14};
        std::vector<double> max_state = {10.0, 10.0, 3.14};

        this->declare_parameter("min_input", min_input);
        this->declare_parameter("max_input", max_input);
        this->declare_parameter("min_state", min_state);
        this->declare_parameter("max_state", max_state);

        // Frame parameters
        this->declare_parameter("robot_frame", "base_link");
        this->declare_parameter("world_frame", "odom");

        // Control rate
        this->declare_parameter("control_rate", 10.0);
    }

    void MPCNode::loadParameters() {
        // Load parameters
        control_rate_ = this->get_parameter("control_rate").as_double();
        robot_frame_ = this->get_parameter("robot_frame").as_string();
        world_frame_ = this->get_parameter("world_frame").as_string();
    }

    void MPCNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        // Extract position and orientation from odometry
        current_state_.x() = msg->pose.pose.position.x;
        current_state_.y() = msg->pose.pose.position.y;

        // Convert quaternion to yaw
        tf2::Quaternion q(
                msg->pose.pose.orientation.x,
                msg->pose.pose.orientation.y,
                msg->pose.pose.orientation.z,
                msg->pose.pose.orientation.w
        );
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        current_state_.z() = yaw;

        has_odom_ = true;
    }

    void MPCNode::pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
        reference_trajectory_.states.clear();
        reference_trajectory_.inputs.clear();

        // Convert path to reference trajectory
        for (const auto& pose_stamped : msg->poses) {
            mpc::State state;
            state.x() = pose_stamped.pose.position.x;
            state.y() = pose_stamped.pose.position.y;

            // Convert quaternion to yaw
            tf2::Quaternion q(
                    pose_stamped.pose.orientation.x,
                    pose_stamped.pose.orientation.y,
                    pose_stamped.pose.orientation.z,
                    pose_stamped.pose.orientation.w
            );
            tf2::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);
            state.z() = yaw;

            reference_trajectory_.states.push_back(state);

            // For simplicity, assume zero input for reference
            reference_trajectory_.inputs.push_back(mpc::ControlInput(0.0, 0.0));
        }

        has_reference_ = true;
        RCLCPP_INFO(this->get_logger(), "Received new path with %zu points", msg->poses.size());
    }

    void MPCNode::controlTimerCallback() {
        if (!has_odom_ || !has_reference_) {
            return;
        }

        // Solve MPC problem
        mpc::ControlInput optimal_input;
        std::vector<mpc::State> predicted_states;

        if (solver_.solve(current_state_, reference_trajectory_, optimal_input, predicted_states)) {
            // Publish control command
            publishCommand(optimal_input);

            // Publish predicted path for visualization
            publishPredictedPath(predicted_states);

            RCLCPP_DEBUG(this->get_logger(),
                         "MPC solved: v=%.2f, ω=%.2f", optimal_input[0], optimal_input[1]);
        } else {
            RCLCPP_WARN(this->get_logger(), "MPC solver failed");
        }
    }

    void MPCNode::publishCommand(const mpc::ControlInput& control_input) {
        auto twist_msg = geometry_msgs::msg::Twist();
        twist_msg.linear.x = control_input[0];  // vx
        twist_msg.linear.y = 0.0;               // vy (for omnidirectional robots)
        twist_msg.angular.z = control_input[1]; // ω

        cmd_vel_pub_->publish(twist_msg);
    }

    void MPCNode::publishPredictedPath(const std::vector<mpc::State>& predicted_states) {
        auto path_msg = nav_msgs::msg::Path();
        path_msg.header.stamp = this->now();
        path_msg.header.frame_id = world_frame_;

        for (const auto& state : predicted_states) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header.stamp = this->now();
            pose.header.frame_id = world_frame_;

            pose.pose.position.x = state.x();
            pose.pose.position.y = state.y();
            pose.pose.position.z = 0.0;

            // Convert yaw to quaternion
            tf2::Quaternion q;
            q.setRPY(0, 0, state.z());
            pose.pose.orientation = tf2::toMsg(q);

            path_msg.poses.push_back(pose);
        }

        predicted_path_pub_->publish(path_msg);
    }

    bool MPCNode::getRobotPose(geometry_msgs::msg::PoseStamped& robot_pose) {
        try {
            geometry_msgs::msg::TransformStamped transform =
                    tf_buffer_->lookupTransform(world_frame_, robot_frame_, tf2::TimePointZero);

            robot_pose.header.stamp = this->now();
            robot_pose.header.frame_id = world_frame_;
            robot_pose.pose.position.x = transform.transform.translation.x;
            robot_pose.pose.position.y = transform.transform.translation.y;
            robot_pose.pose.position.z = transform.transform.translation.z;
            robot_pose.pose.orientation = transform.transform.rotation;

            return true;
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "TF exception: %s", ex.what());
            return false;
        }
    }

}  // namespace mpc_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(mpc_ros::MPCNode)