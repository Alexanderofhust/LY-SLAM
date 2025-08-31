#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include "mpc_osqp_ros2/SystemModel.hpp"
#include "mpc_osqp_ros2/Constraints.hpp"
#include "mpc_osqp_ros2/Types.hpp"

namespace mpc {

/**
 * @class MPCSolver
 * @brief MPC求解器类，使用OSQP-Eigen进行二次规划求解
 */
    class MPCSolver {
    public:
        /**
         * @brief 构造函数
         */
        MPCSolver();

        /**
         * @brief 析构函数
         */
        ~MPCSolver();

        /**
         * @brief 设置MPC求解器配置
         * @param config MPC配置参数
         * @return 是否设置成功
         */
        bool setup(const MPCConfig& config);

        /**
         * @brief 求解MPC问题
         * @param current_state 当前状态
         * @param reference 参考轨迹
         * @param optimal_input 输出的最优控制输入
         * @param predicted_states 输出的预测状态序列（可选）
         * @return 是否求解成功
         */
        bool solve(const State& current_state,
                   const ReferenceTrajectory& reference,
                   ControlInput& optimal_input,
                   std::vector<State>* predicted_states = nullptr);

        /**
         * @brief 更新系统模型
         * @param model 系统模型
         */
        void updateModel(const SystemModel& model);

        /**
         * @brief 更新约束条件
         * @param constraints 约束条件
         */
        void updateConstraints(const Constraints& constraints);

        /**
         * @brief 获取求解时间统计
         * @return 平均求解时间（毫秒）
         */
        double getAverageSolveTime() const;

        /**
         * @brief 重置求解时间统计
         */
        void resetSolveTimeStats();

        /**
         * @brief 获取求解器状态
         * @return 求解器状态字符串
         */
        std::string getSolverStatus() const;

    private:
        /**
         * @brief 构建QP问题的Hessian矩阵
         */
        void constructHessian();

        /**
         * @brief 构建QP问题的梯度向量
         * @param current_state 当前状态
         * @param reference 参考轨迹
         */
        void constructGradient(const State& current_state,
                               const ReferenceTrajectory& reference);

        /**
         * @brief 构建QP问题的约束矩阵
         */
        void constructConstraintMatrix();

        /**
         * @brief 构建QP问题的约束边界
         * @param current_state 当前状态
         */
        void constructConstraintBounds(const State& current_state);

        /**
         * @brief 从QP解中提取控制输入和预测状态
         * @param solution QP解向量
         * @param optimal_input 输出的最优控制输入
         * @param predicted_states 输出的预测状态序列
         */
        void extractSolution(const Eigen::VectorXd& solution,
                             ControlInput& optimal_input,
                             std::vector<State>* predicted_states);

        // OSQP求解器实例
        std::unique_ptr<OsqpEigen::Solver> solver_;

        // MPC配置
        MPCConfig config_;

        // 系统模型
        SystemModel model_;

        // 约束条件
        Constraints constraints_;

        // QP问题矩阵和向量
        Eigen::SparseMatrix<double> hessian_;
        Eigen::SparseMatrix<double> constraint_matrix_;
        Eigen::VectorXd gradient_;
        Eigen::VectorXd lower_bound_;
        Eigen::VectorXd upper_bound_;

        // 系统矩阵序列
        std::vector<Eigen::Matrix3d> A_matrices_;
        std::vector<Eigen::Matrix<double, 3, 2>> B_matrices_;

        // 求解时间统计
        double total_solve_time_{0.0};
        int solve_count_{0};

        // 求解器状态
        mutable std::string solver_status_{"Not initialized"};

        // 标志位
        bool is_initialized_{false};
        bool is_solver_setup_{false};
    };

}  // namespace mpc