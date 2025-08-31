#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "mpc_osqp_ros2/MPCSolver.hpp"
#include "mpc_osqp_ros2/SystemModel.hpp"
#include "mpc_osqp_ros2/Constraints.hpp"

class MPCSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化ROS2（如果需要）
        rclcpp::init(0, nullptr);

        // 设置MPC配置
        config_.prediction_horizon = 10;
        config_.control_horizon = 5;
        config_.time_step = 0.1;

        // 设置权重矩阵
        config_.state_weight << 10.0, 0.0, 0.0,
                0.0, 10.0, 0.0,
                0.0, 0.0, 5.0;
        config_.input_weight << 1.0, 0.0,
                0.0, 1.0;
        config_.input_rate_weight << 0.1, 0.0,
                0.0, 0.1;

        // 初始化求解器
        solver_.setup(config_);

        // 设置系统模型
        mpc::ModelParameters model_params;
        model_params.wheel_radius = 0.1;
        model_params.wheel_base = 0.5;
        model_.setParameters(model_params);
        solver_.updateModel(model_);

        // 设置约束
        constraints_.setInputConstraints(
                mpc::ControlInput(-0.5, -1.0),
                mpc::ControlInput(0.5, 1.0)
        );
        constraints_.setStateConstraints(
                mpc::State(-10.0, -10.0, -3.14),
                mpc::State(10.0, 10.0, 3.14)
        );
        solver_.updateConstraints(constraints_);
    }

    void TearDown() override {
        rclcpp::shutdown();
    }

    mpc::MPCSolver solver_;
    mpc::MPCConfig config_;
    mpc::SystemModel model_;
    mpc::Constraints constraints_;
};

// 测试求解器初始化
TEST_F(MPCSolverTest, Initialization) {
EXPECT_TRUE(solver_.getSolverStatus().find("Initialized") != std::string::npos);
}

// 测试求解器配置验证
TEST_F(MPCSolverTest, ConfigValidation) {
mpc::MPCConfig invalid_config;
invalid_config.prediction_horizon = 0; // 无效的预测时域

mpc::MPCSolver test_solver;
EXPECT_FALSE(test_solver.setup(invalid_config));
}

// 测试参考轨迹验证
TEST_F(MPCSolverTest, ReferenceValidation) {
mpc::State current_state(0.0, 0.0, 0.0);
mpc::ReferenceTrajectory empty_reference;
mpc::ControlInput optimal_input;

// 测试空参考轨迹
EXPECT_FALSE(solver_.solve(current_state, empty_reference, optimal_input));

// 测试短参考轨迹
mpc::ReferenceTrajectory short_reference;
short_reference.states.resize(5); // 小于预测时域
EXPECT_FALSE(solver_.solve(current_state, short_reference, optimal_input));
}

// 测试MPC求解
TEST_F(MPCSolverTest, MPCSolve) {
// 创建参考轨迹（直线运动）
mpc::ReferenceTrajectory reference;
for (int i = 0; i < config_.prediction_horizon; ++i) {
double t = i * config_.time_step;
reference.states.emplace_back(t, 0.0, 0.0);
reference.inputs.emplace_back(1.0, 0.0);
}

mpc::State current_state(0.0, 0.0, 0.0);
mpc::ControlInput optimal_input;
std::vector<mpc::State> predicted_states;

// 测试求解
EXPECT_TRUE(solver_.solve(current_state, reference, optimal_input, &predicted_states));

// 验证求解结果
EXPECT_NEAR(optimal_input(0), 1.0, 0.1); // 线速度接近1.0
EXPECT_NEAR(optimal_input(1), 0.0, 0.1); // 角速度接近0.0

// 验证预测状态序列
EXPECT_EQ(predicted_states.size(), config_.prediction_horizon);

// 验证第一个预测状态应该接近当前状态
EXPECT_NEAR(predicted_states[0](0), current_state(0), 0.01);
EXPECT_NEAR(predicted_states[0](1), current_state(1), 0.01);
EXPECT_NEAR(predicted_states[0](2), current_state(2), 0.01);
}

// 测试约束违反
TEST_F(MPCSolverTest, ConstraintViolation) {
// 创建参考轨迹（超出约束范围）
mpc::ReferenceTrajectory reference;
for (int i = 0; i < config_.prediction_horizon; ++i) {
reference.states.emplace_back(20.0, 20.0, 0.0); // 超出状态约束
reference.inputs.emplace_back(2.0, 2.0); // 超出输入约束
}

mpc::State current_state(0.0, 0.0, 0.0);
mpc::ControlInput optimal_input;

// 求解应该成功，但输入应该被约束在合理范围内
EXPECT_TRUE(solver_.solve(current_state, reference, optimal_input));

// 验证输入是否被约束
EXPECT_LE(optimal_input(0), 0.5); // 线速度不超过0.5
EXPECT_GE(optimal_input(0), -0.5); // 线速度不低于-0.5
EXPECT_LE(optimal_input(1), 1.0); // 角速度不超过1.0
EXPECT_GE(optimal_input(1), -1.0); // 角速度不低于-1.0
}

// 测试求解时间统计
TEST_F(MPCSolverTest, SolveTimeStats) {
// 创建参考轨迹
mpc::ReferenceTrajectory reference;
for (int i = 0; i < config_.prediction_horizon; ++i) {
reference.states.emplace_back(i * 0.1, 0.0, 0.0);
reference.inputs.emplace_back(1.0, 0.0);
}

mpc::State current_state(0.0, 0.0, 0.0);
mpc::ControlInput optimal_input;

// 多次求解以收集统计信息
for (int i = 0; i < 5; ++i) {
EXPECT_TRUE(solver_.solve(current_state, reference, optimal_input));
}

// 验证求解时间统计
double avg_time = solver_.getAverageSolveTime();
EXPECT_GT(avg_time, 0.0); // 平均求解时间应该大于0
EXPECT_LT(avg_time, 1000.0); // 平均求解时间应该小于1秒（合理范围内）

// 重置统计信息
solver_.resetSolveTimeStats();
EXPECT_EQ(solver_.getAverageSolveTime(), 0.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}