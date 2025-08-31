#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "mpc_osqp_ros2/SystemModel.hpp"

class SystemModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置模型参数
        params_.wheel_radius = 0.1;
        params_.wheel_base = 0.5;
        params_.mass = 10.0;
        params_.moment_of_inertia = 2.0;

        model_.setParameters(params_);
    }

    mpc::SystemModel model_;
    mpc::ModelParameters params_;
};

// 测试模型参数验证
TEST_F(SystemModelTest, ParameterValidation) {
mpc::ModelParameters invalid_params;
invalid_params.wheel_radius = -0.1; // 无效的轮子半径

mpc::SystemModel test_model;
EXPECT_THROW(test_model.setParameters(invalid_params), std::invalid_argument);
}

// 测试连续时间动力学
TEST_F(SystemModelTest, ContinuousDynamics) {
mpc::State state(0.0, 0.0, 0.0); // 原点，朝向0度
mpc::ControlInput input(1.0, 0.5); // 线速度1.0，角速度0.5

mpc::State derivative = model_.continuousDynamics(state, input);

// 验证导数计算
EXPECT_NEAR(derivative(0), 1.0, 1e-6); // ẋ = v * cos(θ) = 1.0 * 1.0 = 1.0
EXPECT_NEAR(derivative(1), 0.0, 1e-6); // ẏ = v * sin(θ) = 1.0 * 0.0 = 0.0
EXPECT_NEAR(derivative(2), 0.5, 1e-6); // θ̇ = ω = 0.5
}

// 测试欧拉离散化
TEST_F(SystemModelTest, EulerDiscretization) {
mpc::State state(0.0, 0.0, 0.0);
mpc::ControlInput input(1.0, 0.5);
double dt = 0.1;

mpc::State next_state = model_.discreteDynamicsEuler(state, input, dt);

// 验证离散化结果
EXPECT_NEAR(next_state(0), 0.1, 1e-6); // x = 0.0 + 1.0 * 0.1 = 0.1
EXPECT_NEAR(next_state(1), 0.0, 1e-6); // y = 0.0 + 0.0 * 0.1 = 0.0
EXPECT_NEAR(next_state(2), 0.05, 1e-6); // θ = 0.0 + 0.5 * 0.1 = 0.05
}

// 测试龙格-库塔离散化
TEST_F(SystemModelTest, RK4Discretization) {
mpc::State state(0.0, 0.0, 0.0);
mpc::ControlInput input(1.0, 0.5);
double dt = 0.1;

mpc::State next_state = model_.discreteDynamicsRK4(state, input, dt);

// 验证离散化结果（应该比欧拉方法更精确）
EXPECT_NEAR(next_state(0), 0.1, 1e-6);
EXPECT_NEAR(next_state(1), 0.0, 1e-6);
EXPECT_NEAR(next_state(2), 0.05, 1e-6);
}

// 测试系统线性化
TEST_F(SystemModelTest, Linearization) {
mpc::State state(0.0, 0.0, 0.0);
mpc::ControlInput input(1.0, 0.5);
double dt = 0.1;

auto [A, B] = model_.linearize(state, input, dt);

// 验证系统矩阵A
EXPECT_NEAR(A(0, 0), 1.0, 1e-6);
EXPECT_NEAR(A(0, 1), 0.0, 1e-6);
EXPECT_NEAR(A(0, 2), 0.0, 1e-6); // 对于小角度，线性化后应为0

EXPECT_NEAR(A(1, 0), 0.0, 1e-6);
EXPECT_NEAR(A(1, 1), 1.0, 1e-6);
EXPECT_NEAR(A(1, 2), 0.1, 1e-6); // 对于小角度，线性化后应为v*dt

EXPECT_NEAR(A(2, 0), 0.0, 1e-6);
EXPECT_NEAR(A(2, 1), 0.0, 1e-6);
EXPECT_NEAR(A(2, 2), 1.0, 1e-6);

// 验证输入矩阵B
EXPECT_NEAR(B(0, 0), 0.1, 1e-6); // cos(θ)*dt = 1.0 * 0.1 = 0.1
EXPECT_NEAR(B(0, 1), 0.0, 1e-6);

EXPECT_NEAR(B(1, 0), 0.0, 1e-6); // sin(θ)*dt = 0.0 * 0.1 = 0.0
EXPECT_NEAR(B(1, 1), 0.0, 1e-6);

EXPECT_NEAR(B(2, 0), 0.0, 1e-6);
EXPECT_NEAR(B(2, 1), 0.1, 1e-6); // dt = 0.1
}

// 测试状态序列预测
TEST_F(SystemModelTest, StatePrediction) {
mpc::State initial_state(0.0, 0.0, 0.0);
std::vector<mpc::ControlInput> input_sequence(10, mpc::ControlInput(1.0, 0.1));
double dt = 0.1;

std::vector<mpc::State> predicted_states = model_.predictStateSequence(
        initial_state, input_sequence, dt);

// 验证预测结果
EXPECT_EQ(predicted_states.size(), input_sequence.size() + 1);

// 验证最后一个状态
EXPECT_GT(predicted_states.back()(0), 0.0); // x应该增加
EXPECT_NEAR(predicted_states.back()(1), 0.0, 0.1); // y应该接近0（小角度近似）
EXPECT_GT(predicted_states.back()(2), 0.0); // θ应该增加
}

// 测试状态误差计算
TEST_F(SystemModelTest, StateError) {
mpc::State state1(1.0, 2.0, 0.5);
mpc::State state2(0.5, 1.0, 0.2);

mpc::State error = model_.computeStateError(state1, state2);

// 验证误差计算
EXPECT_NEAR(error(0), 0.5, 1e-6); // 1.0 - 0.5 = 0.5
EXPECT_NEAR(error(1), 1.0, 1e-6); // 2.0 - 1.0 = 1.0
EXPECT_NEAR(error(2), 0.3, 1e-6); // 0.5 - 0.2 = 0.3

// 测试角度规范化
mpc::State state3(0.0, 0.0, 3.0 * M_PI);
mpc::State state4(0.0, 0.0, 0.0);

mpc::State angle_error = model_.computeStateError(state3, state4);
EXPECT_NEAR(angle_error(2), -M_PI, 1e-6); // 应该规范化到[-π, π]范围
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}