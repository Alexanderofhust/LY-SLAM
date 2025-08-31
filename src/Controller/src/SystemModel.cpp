#include "mpc_osqp_ros2/SystemModel.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace mpc {

    SystemModel::SystemModel() {
        // 设置默认模型参数
        resetParameters();
    }

    SystemModel::~SystemModel() {
        // 清理资源（如果需要）
    }

    void SystemModel::setParameters(const ModelParameters& params) {
        params_ = params;

        // 验证参数
        if (!validateParameters()) {
            throw std::invalid_argument("Invalid model parameters");
        }
    }

    const ModelParameters& SystemModel::getParameters() const {
        return params_;
    }

    State SystemModel::continuousDynamics(const State& state, const ControlInput& input) const {
        // 对于差速驱动机器人，连续时间动力学为：
        // ẋ = v * cos(θ)
        // ẏ = v * sin(θ)
        // θ̇ = ω

        State state_derivative;
        state_derivative(0) = input(0) * std::cos(state(2));  // ẋ = v * cos(θ)
        state_derivative(1) = input(0) * std::sin(state(2));  // ẏ = v * sin(θ)
        state_derivative(2) = input(1);                       // θ̇ = ω

        return state_derivative;
    }

    State SystemModel::discreteDynamicsEuler(const State& state, const ControlInput& input, double dt) const {
        // 使用欧拉积分进行离散化：x_{k+1} = x_k + f(x_k, u_k) * dt
        State next_state = state + continuousDynamics(state, input) * dt;

        // 规范化角度到 [-π, π] 范围
        next_state(2) = std::fmod(next_state(2) + M_PI, 2 * M_PI) - M_PI;

        return next_state;
    }

    State SystemModel::discreteDynamicsRK4(const State& state, const ControlInput& input, double dt) const {
        // 使用4阶龙格-库塔方法进行离散化
        State k1 = continuousDynamics(state, input);
        State k2 = continuousDynamics(state + k1 * (dt / 2.0), input);
        State k3 = continuousDynamics(state + k2 * (dt / 2.0), input);
        State k4 = continuousDynamics(state + k3 * dt, input);

        State next_state = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0);

        // 规范化角度到 [-π, π] 范围
        next_state(2) = std::fmod(next_state(2) + M_PI, 2 * M_PI) - M_PI;

        return next_state;
    }

    std::pair<Eigen::Matrix3d, Eigen::Matrix<double, 3, 2>> SystemModel::linearize(
            const State& state, const ControlInput& input, double dt) const {

        // 计算连续时间雅可比矩阵
        Eigen::Matrix3d A_cont = computeStateJacobian(state, input);
        Eigen::Matrix<double, 3, 2> B_cont = computeInputJacobian(state, input);

        // 使用矩阵指数近似离散化（一阶近似）
        Eigen::Matrix3d A_disc = Eigen::Matrix3d::Identity() + A_cont * dt;
        Eigen::Matrix<double, 3, 2> B_disc = B_cont * dt;

        return std::make_pair(A_disc, B_disc);
    }

    std::vector<State> SystemModel::predictStateSequence(
            const State& initial_state,
            const std::vector<ControlInput>& input_sequence,
            double dt) const {

        std::vector<State> state_sequence;
        state_sequence.reserve(input_sequence.size() + 1);

        State current_state = initial_state;
        state_sequence.push_back(current_state);

        for (const auto& input : input_sequence) {
            current_state = discreteDynamicsRK4(current_state, input, dt);
            state_sequence.push_back(current_state);
        }

        return state_sequence;
    }

    State SystemModel::computeStateError(const State& state1, const State& state2) const {
        State error = state1 - state2;

        // 处理角度误差（确保在 [-π, π] 范围内）
        error(2) = std::fmod(error(2) + M_PI, 2 * M_PI) - M_PI;

        return error;
    }

    ControlInput SystemModel::computeInputError(const ControlInput& input1, const ControlInput& input2) const {
        return input1 - input2;
    }

    bool SystemModel::validateParameters() const {
        // 检查参数是否有效
        if (params_.wheel_radius <= 0) {
            std::cerr << "Invalid wheel radius: " << params_.wheel_radius << std::endl;
            return false;
        }

        if (params_.wheel_base <= 0) {
            std::cerr << "Invalid wheel base: " << params_.wheel_base << std::endl;
            return false;
        }

        if (params_.mass <= 0) {
            std::cerr << "Invalid mass: " << params_.mass << std::endl;
            return false;
        }

        if (params_.moment_of_inertia <= 0) {
            std::cerr << "Invalid moment of inertia: " << params_.moment_of_inertia << std::endl;
            return false;
        }

        return true;
    }

    void SystemModel::resetParameters() {
        // 重置为默认参数
        params_.wheel_radius = 0.1;      // 轮子半径 (m)
        params_.wheel_base = 0.5;        // 轮距 (m)
        params_.mass = 10.0;             // 质量 (kg)
        params_.moment_of_inertia = 2.0; // 转动惯量 (kg·m²)
    }

    Eigen::Matrix3d SystemModel::computeStateJacobian(const State& state, const ControlInput& input) const {
        // 计算状态雅可比矩阵：∂f/∂x
        // 对于差速驱动机器人：
        // f1 = v * cos(θ) → ∂f1/∂θ = -v * sin(θ)
        // f2 = v * sin(θ) → ∂f2/∂θ = v * cos(θ)
        // f3 = ω → ∂f3/∂x = ∂f3/∂y = ∂f3/∂θ = 0

        Eigen::Matrix3d jacobian = Eigen::Matrix3d::Zero();

        jacobian(0, 2) = -input(0) * std::sin(state(2));  // ∂f1/∂θ
        jacobian(1, 2) = input(0) * std::cos(state(2));   // ∂f2/∂θ
        jacobian(2, 2) = 0.0;                             // ∂f3/∂θ

        return jacobian;
    }

    Eigen::Matrix<double, 3, 2> SystemModel::computeInputJacobian(const State& state, const ControlInput& input) const {
        // 计算输入雅可比矩阵：∂f/∂u
        // 对于差速驱动机器人：
        // f1 = v * cos(θ) → ∂f1/∂v = cos(θ), ∂f1/∂ω = 0
        // f2 = v * sin(θ) → ∂f2/∂v = sin(θ), ∂f2/∂ω = 0
        // f3 = ω → ∂f3/∂v = 0, ∂f3/∂ω = 1

        Eigen::Matrix<double, 3, 2> jacobian;

        jacobian(0, 0) = std::cos(state(2));  // ∂f1/∂v
        jacobian(0, 1) = 0.0;                 // ∂f1/∂ω

        jacobian(1, 0) = std::sin(state(2));  // ∂f2/∂v
        jacobian(1, 1) = 0.0;                 // ∂f2/∂ω

        jacobian(2, 0) = 0.0;                 // ∂f3/∂v
        jacobian(2, 1) = 1.0;                 // ∂f3/∂ω

        return jacobian;
    }

    State SystemModel::rk4Step(const State& state, const ControlInput& input, double dt, const State& k) const {
        return state + k * dt;
    }

}  // namespace mpc