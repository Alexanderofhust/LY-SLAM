#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "mpc_osqp_ros2/Types.hpp"

namespace mpc {

/**
 * @class SystemModel
 * @brief 定义机器人系统模型，包括运动学模型和动力学模型
 *
 * 这个类提供了机器人系统的数学模型，包括：
 * 1. 连续时间和离散时间的系统动力学
 * 2. 系统线性化方法
 * 3. 状态预测功能
 */
    class SystemModel {
    public:
        /**
         * @brief 构造函数
         */
        SystemModel();

        /**
         * @brief 析构函数
         */
        ~SystemModel();

        /**
         * @brief 设置模型参数
         * @param params 模型参数结构体
         */
        void setParameters(const ModelParameters& params);

        /**
         * @brief 获取模型参数
         * @return 模型参数结构体
         */
        const ModelParameters& getParameters() const;

        /**
         * @brief 连续时间系统动力学
         * @param state 当前状态 [x, y, θ]^T
         * @param input 控制输入 [v, ω]^T
         * @return 状态导数 [ẋ, ẏ, θ̇]^T
         */
        State continuousDynamics(const State& state, const ControlInput& input) const;

        /**
         * @brief 离散时间系统动力学（欧拉积分）
         * @param state 当前状态 [x, y, θ]^T
         * @param input 控制输入 [v, ω]^T
         * @param dt 时间步长
         * @return 下一时刻状态 [x_{k+1}, y_{k+1}, θ_{k+1}]^T
         */
        State discreteDynamicsEuler(const State& state, const ControlInput& input, double dt) const;

        /**
         * @brief 离散时间系统动力学（龙格-库塔积分）
         * @param state 当前状态 [x, y, θ]^T
         * @param input 控制输入 [v, ω]^T
         * @param dt 时间步长
         * @return 下一时刻状态 [x_{k+1}, y_{k+1}, θ_{k+1}]^T
         */
        State discreteDynamicsRK4(const State& state, const ControlInput& input, double dt) const;

        /**
         * @brief 系统线性化（计算雅可比矩阵）
         * @param state 当前状态 [x, y, θ]^T
         * @param input 控制输入 [v, ω]^T
         * @param dt 时间步长
         * @return 包含系统矩阵A和输入矩阵B的pair
         */
        std::pair<Eigen::Matrix3d, Eigen::Matrix<double, 3, 2>> linearize(
                const State& state, const ControlInput& input, double dt) const;

        /**
         * @brief 预测状态序列
         * @param initial_state 初始状态
         * @param input_sequence 输入序列
         * @param dt 时间步长
         * @return 预测的状态序列
         */
        std::vector<State> predictStateSequence(
                const State& initial_state,
                const std::vector<ControlInput>& input_sequence,
                double dt) const;

        /**
         * @brief 计算状态误差
         * @param state1 状态1
         * @param state2 状态2
         * @return 状态误差向量
         */
        State computeStateError(const State& state1, const State& state2) const;

        /**
         * @brief 计算输入误差
         * @param input1 输入1
         * @param input2 输入2
         * @return 输入误差向量
         */
        ControlInput computeInputError(const ControlInput& input1, const ControlInput& input2) const;

        /**
         * @brief 检查模型参数是否有效
         * @return 如果参数有效返回true，否则返回false
         */
        bool validateParameters() const;

        /**
         * @brief 重置模型参数为默认值
         */
        void resetParameters();

    private:
        // 模型参数
        ModelParameters params_;

        /**
         * @brief 计算系统雅可比矩阵（状态导数关于状态的偏导数）
         * @param state 当前状态
         * @param input 控制输入
         * @return 状态雅可比矩阵 (3x3)
         */
        Eigen::Matrix3d computeStateJacobian(const State& state, const ControlInput& input) const;

        /**
         * @brief 计算输入雅可比矩阵（状态导数关于输入的偏导数）
         * @param state 当前状态
         * @param input 控制输入
         * @return 输入雅可比矩阵 (3x2)
         */
        Eigen::Matrix<double, 3, 2> computeInputJacobian(const State& state, const ControlInput& input) const;

        /**
         * @brief 内部RK4积分步骤
         * @param state 当前状态
         * @param input 控制输入
         * @param dt 时间步长
         * @param k 龙格-库塔系数
         * @return 积分步骤结果
         */
        State rk4Step(const State& state, const ControlInput& input, double dt, const State& k) const;
    };

}  // namespace mpc