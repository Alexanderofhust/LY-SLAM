#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cstddef>

namespace mpc {

/**
 * @brief 机器人状态向量 [x, y, θ]^T
 *
 * 表示机器人在全局坐标系中的位姿：
 * - x: x坐标 (米)
 * - y: y坐标 (米)
 * - θ: 朝向角 (弧度)
 */
    using State = Eigen::Vector3d;

/**
 * @brief 控制输入向量 [v, ω]^T
 *
 * 表示机器人的控制输入：
 * - v: 线速度 (米/秒)
 * - ω: 角速度 (弧度/秒)
 */
    using ControlInput = Eigen::Vector2d;

/**
 * @brief 模型参数结构体
 *
 * 包含机器人的物理参数和运动学参数
 */
    struct ModelParameters {
        double wheel_radius{0.1};        ///< 轮子半径 (米)
        double wheel_base{0.5};          ///< 轮距 (米)，对于差速驱动机器人
        double track_width{0.4};         ///< 轮距 (米)，对于全向移动机器人
        double mass{10.0};               ///< 质量 (千克)
        double moment_of_inertia{2.0};   ///< 转动惯量 (千克·平方米)

        // 舵轮特定参数
        double steering_offset{0.0};     ///< 转向偏移 (弧度)
        double max_steering_angle{M_PI/4}; ///< 最大转向角 (弧度)

        // 动力学参数
        double friction_coefficient{0.1}; ///< 摩擦系数
        double damping_coefficient{0.05}; ///< 阻尼系数

        /**
         * @brief 检查参数是否有效
         * @return 如果所有参数都有效返回true，否则返回false
         */
        bool isValid() const {
            return wheel_radius > 0 && wheel_base > 0 && track_width > 0 &&
                   mass > 0 && moment_of_inertia > 0 &&
                   max_steering_angle > 0 && max_steering_angle <= M_PI &&
                   friction_coefficient >= 0 && damping_coefficient >= 0;
        }
    };

/**
 * @brief MPC配置结构体
 *
 * 包含MPC控制器的配置参数
 */
    struct MPCConfig {
        int prediction_horizon{20};    ///< 预测时域（步数）
        int control_horizon{10};       ///< 控制时域（步数）
        double time_step{0.1};            ///< 时间步长（秒）

        // 权重矩阵
        Eigen::Matrix3d state_weight{Eigen::Matrix3d::Identity()};      ///< 状态权重矩阵
        Eigen::Matrix2d input_weight{0.1 * Eigen::Matrix2d::Identity()}; ///< 输入权重矩阵
        Eigen::Matrix2d input_rate_weight{0.01 * Eigen::Matrix2d::Identity()}; ///< 输入变化率权重矩阵

        // 终端代价权重
        Eigen::Matrix3d terminal_weight{5.0 * Eigen::Matrix3d::Identity()}; ///< 终端状态权重矩阵

        // 松弛变量权重（用于软约束）
        double state_constraint_weight{1e3}; ///< 状态约束违反权重
        double input_constraint_weight{1e3}; ///< 输入约束违反权重

        // 求解器设置
        int max_iterations{1000};          ///< 最大迭代次数
        double eps_abs{1e-4};              ///< 绝对容忍度
        double eps_rel{1e-4};              ///< 相对容忍度
        double rho{0.1};                   ///< ADMM参数rho
        double sigma{1e-6};                ///< ADMM参数sigma
        double alpha{1.6};                 ///< ADMM参数alpha（过松弛参数）

        /**
         * @brief 检查配置是否有效
         * @return 如果所有配置都有效返回true，否则返回false
         */
        bool isValid() const {
            return prediction_horizon > 0 && control_horizon > 0 &&
                   control_horizon <= prediction_horizon &&
                   time_step > 0 &&
                   state_weight.rows() == 3 && state_weight.cols() == 3 &&
                   input_weight.rows() == 2 && input_weight.cols() == 2 &&
                   input_rate_weight.rows() == 2 && input_rate_weight.cols() == 2 &&
                   terminal_weight.rows() == 3 && terminal_weight.cols() == 3 &&
                   state_constraint_weight >= 0 && input_constraint_weight >= 0 &&
                   max_iterations > 0 && eps_abs > 0 && eps_rel > 0 &&
                   rho > 0 && sigma > 0 && alpha > 0 && alpha <= 2.0;
        }
    };

/**
 * @brief 参考轨迹结构体
 *
 * 包含MPC跟踪的参考轨迹信息
 */
    struct ReferenceTrajectory {
        std::vector<State> states;          ///< 参考状态序列
        std::vector<ControlInput> inputs;   ///< 参考输入序列
        std::vector<double> timestamps;     ///< 时间戳序列（可选）

        /**
         * @brief 检查参考轨迹是否有效
         * @param prediction_horizon 预测时域
         * @return 如果参考轨迹有效返回true，否则返回false
         */
        bool isValid(size_t prediction_horizon) const {
            return !states.empty() && states.size() >= prediction_horizon &&
                   (inputs.empty() || inputs.size() >= prediction_horizon);
        }

        /**
         * @brief 获取参考轨迹长度
         * @return 参考轨迹长度
         */
        size_t size() const {
            return states.size();
        }

        /**
         * @brief 检查参考轨迹是否为空
         * @return 如果参考轨迹为空返回true，否则返回false
         */
        bool empty() const {
            return states.empty();
        }
    };

/**
 * @brief 约束边界结构体
 *
 * 包含状态和输入的约束边界
 */
    struct ConstraintBounds {
        State min_state;                    ///< 状态最小值 [x_min, y_min, θ_min]^T
        State max_state;                    ///< 状态最大值 [x_max, y_max, θ_max]^T
        ControlInput min_input;             ///< 输入最小值 [v_min, ω_min]^T
        ControlInput max_input;             ///< 输入最大值 [v_max, ω_max]^T
        ControlInput min_input_rate;        ///< 输入变化率最小值 [Δv_min, Δω_min]^T
        ControlInput max_input_rate;        ///< 输入变化率最大值 [Δv_max, Δω_max]^T

        /**
         * @brief 检查约束边界是否有效
         * @return 如果所有约束边界都有效返回true，否则返回false
         */
        bool isValid() const {
            for (int i = 0; i < 3; ++i) {
                if (min_state(i) > max_state(i)) return false;
            }
            for (int i = 0; i < 2; ++i) {
                if (min_input(i) > max_input(i)) return false;
                if (min_input_rate(i) > max_input_rate(i)) return false;
            }
            return true;
        }

        /**
         * @brief 设置对称约束
         * @param state_bound 状态对称边界 [x_bound, y_bound, θ_bound]^T
         * @param input_bound 输入对称边界 [v_bound, ω_bound]^T
         * @param input_rate_bound 输入变化率对称边界 [Δv_bound, Δω_bound]^T
         */
        void setSymmetricBounds(const State& state_bound,
                                const ControlInput& input_bound,
                                const ControlInput& input_rate_bound) {
            min_state = -state_bound;
            max_state = state_bound;
            min_input = -input_bound;
            max_input = input_bound;
            min_input_rate = -input_rate_bound;
            max_input_rate = input_rate_bound;
        }
    };

/**
 * @brief MPC求解结果结构体
 *
 * 包含MPC求解的结果信息
 */
    struct MPCResult {
        bool success{false};                ///< 求解是否成功
        ControlInput optimal_input;         ///< 最优控制输入
        std::vector<State> predicted_states; ///< 预测状态序列
        std::vector<ControlInput> predicted_inputs; ///< 预测输入序列
        double solve_time{0.0};             ///< 求解时间（毫秒）
        int iterations{0};                  ///< 迭代次数
        std::string status_message;         ///< 状态消息

        /**
         * @brief 检查结果是否有效
         * @return 如果结果有效返回true，否则返回false
         */
        bool isValid() const {
            return success && !predicted_states.empty() &&
                   predicted_states.size() == predicted_inputs.size() + 1;
        }

        /**
         * @brief 重置结果
         */
        void reset() {
            success = false;
            optimal_input.setZero();
            predicted_states.clear();
            predicted_inputs.clear();
            solve_time = 0.0;
            iterations = 0;
            status_message.clear();
        }
    };

/**
 * @brief 机器人类型枚举
 *
 * 定义支持的机器人类型
 */
    enum class RobotType {
        DIFFERENTIAL_DRIVE,    ///< 差速驱动机器人
        OMNI_DIRECTIONAL,      ///< 全向移动机器人
        ACKERMANN_STEERING,    ///< 阿克曼转向机器人
        MECANUM_WHEEL          ///< 麦克纳姆轮机器人
    };

/**
 * @brief 控制器模式枚举
 *
 * 定义控制器的运行模式
 */
    enum class ControllerMode {
        TRAJECTORY_TRACKING,   ///< 轨迹跟踪模式
        POINT_STABILIZATION,   ///< 点镇定模式
        PATH_FOLLOWING,        ///< 路径跟随模式
        VELOCITY_CONTROL       ///< 速度控制模式
    };

/**
 * @brief 求解器状态枚举
 *
 * 定义MPC求解器的状态
 */
    enum class SolverStatus {
        NOT_INITIALIZED,       ///< 未初始化
        INITIALIZED,           ///< 已初始化
        OPTIMAL,               ///< 找到最优解
        SUBOPTIMAL,            ///< 找到次优解
        MAX_ITERATIONS,        ///< 达到最大迭代次数
        PRIMAL_INFEASIBLE,     ///< 原始问题不可行
        DUAL_INFEASIBLE,       ///< 对偶问题不可行
        NUMERICAL_ISSUES,      ///< 数值问题
        SOLVER_ERROR           ///< 求解器错误
    };

/**
 * @brief 将求解器状态转换为字符串
 * @param status 求解器状态
 * @return 状态字符串
 */
    inline std::string solverStatusToString(SolverStatus status) {
        switch (status) {
            case SolverStatus::NOT_INITIALIZED: return "Not Initialized";
            case SolverStatus::INITIALIZED: return "Initialized";
            case SolverStatus::OPTIMAL: return "Optimal";
            case SolverStatus::SUBOPTIMAL: return "Suboptimal";
            case SolverStatus::MAX_ITERATIONS: return "Max Iterations Reached";
            case SolverStatus::PRIMAL_INFEASIBLE: return "Primal Infeasible";
            case SolverStatus::DUAL_INFEASIBLE: return "Dual Infeasible";
            case SolverStatus::NUMERICAL_ISSUES: return "Numerical Issues";
            case SolverStatus::SOLVER_ERROR: return "Solver Error";
            default: return "Unknown";
        }
    }

/**
 * @brief 将控制器模式转换为字符串
 * @param mode 控制器模式
 * @return 模式字符串
 */
    inline std::string controllerModeToString(ControllerMode mode) {
        switch (mode) {
            case ControllerMode::TRAJECTORY_TRACKING: return "Trajectory Tracking";
            case ControllerMode::POINT_STABILIZATION: return "Point Stabilization";
            case ControllerMode::PATH_FOLLOWING: return "Path Following";
            case ControllerMode::VELOCITY_CONTROL: return "Velocity Control";
            default: return "Unknown";
        }
    }

/**
 * @brief 将机器人类型转换为字符串
 * @param type 机器人类型
 * @return 类型字符串
 */
    inline std::string robotTypeToString(RobotType type) {
        switch (type) {
            case RobotType::DIFFERENTIAL_DRIVE: return "Differential Drive";
            case RobotType::OMNI_DIRECTIONAL: return "Omni-directional";
            case RobotType::ACKERMANN_STEERING: return "Ackermann Steering";
            case RobotType::MECANUM_WHEEL: return "Mecanum Wheel";
            default: return "Unknown";
        }
    }

}  // namespace mpc