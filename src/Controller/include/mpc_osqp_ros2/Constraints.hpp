#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "mpc_osqp_ros2/Types.hpp"

namespace mpc {

/**
 * @class Constraints
 * @brief 处理MPC问题中的约束条件，包括状态约束、输入约束和输入速率约束
 */
    class Constraints {
    public:
        /**
         * @brief 构造函数
         */
        Constraints();

        /**
         * @brief 析构函数
         */
        ~Constraints();

        /**
         * @brief 设置输入约束
         * @param min_input 最小输入向量 [v_min, ω_min]
         * @param max_input 最大输入向量 [v_max, ω_max]
         */
        void setInputConstraints(const ControlInput& min_input, const ControlInput& max_input);

        /**
         * @brief 设置状态约束
         * @param min_state 最小状态向量 [x_min, y_min, θ_min]
         * @param max_state 最大状态向量 [x_max, y_max, θ_max]
         */
        void setStateConstraints(const State& min_state, const State& max_state);

        /**
         * @brief 设置输入速率约束
         * @param min_rate 最小输入变化率 [Δv_min, Δω_min]
         * @param max_rate 最大输入变化率 [Δv_max, Δω_max]
         */
        void setInputRateConstraints(const ControlInput& min_rate, const ControlInput& max_rate);

        /**
         * @brief 获取最小输入约束
         * @return 最小输入约束向量
         */
        ControlInput getMinInput() const;

        /**
         * @brief 获取最大输入约束
         * @return 最大输入约束向量
         */
        ControlInput getMaxInput() const;

        /**
         * @brief 获取最小状态约束
         * @return 最小状态约束向量
         */
        State getMinState() const;

        /**
         * @brief 获取最大状态约束
         * @return 最大状态约束向量
         */
        State getMaxState() const;

        /**
         * @brief 获取最小输入变化率约束
         * @return 最小输入变化率约束向量
         */
        const ControlInput& getMinInputRate() const;

        /**
         * @brief 获取最大输入变化率约束
         * @return 最大输入变化率约束向量
         */
        const ControlInput& getMaxInputRate() const;

        /**
         * @brief 检查输入约束是否已设置
         * @return 如果输入约束已设置返回true，否则返回false
         */
        bool hasInputConstraints() const;

        /**
         * @brief 检查状态约束是否已设置
         * @return 如果状态约束已设置返回true，否则返回false
         */
        bool hasStateConstraints() const;

        /**
         * @brief 检查输入速率约束是否已设置
         * @return 如果输入速率约束已设置返回true，否则返回false
         */
        bool hasInputRateConstraints() const;

        /**
         * @brief 构建约束矩阵和边界向量
         * @param prediction_horizon 预测时域
         * @param control_horizon 控制时域
         * @param state_dim 状态维度
         * @param input_dim 输入维度
         * @param constraint_matrix 输出的约束矩阵（稀疏矩阵）
         * @param lower_bound 输出的下界向量
         * @param upper_bound 输出的上界向量
         * @param initial_state 初始状态（用于设置初始状态约束）
         */
        void constructConstraints(
                size_t prediction_horizon,
                size_t control_horizon,
                size_t state_dim,
                size_t input_dim,
                Eigen::SparseMatrix<double>& constraint_matrix,
                Eigen::VectorXd& lower_bound,
                Eigen::VectorXd& upper_bound,
                const State& initial_state) const;

        /**
         * @brief 重置所有约束
         */
        void reset();

    private:
        // 输入约束
        ControlInput min_input_;
        ControlInput max_input_;
        bool has_input_constraints_{false};

        // 状态约束
        State min_state_;
        State max_state_;
        bool has_state_constraints_{false};

        // 输入速率约束
        ControlInput min_input_rate_;
        ControlInput max_input_rate_;
        bool has_input_rate_constraints_{false};

        /**
         * @brief 添加输入约束到约束矩阵和边界向量
         * @param num_states 状态变量数量
         * @param num_inputs 输入变量数量
         * @param constraint_matrix 约束矩阵（稀疏矩阵）
         * @param lower_bound 下界向量
         * @param upper_bound 上界向量
         * @param triplets 三元组列表（用于构建稀疏矩阵）
         */
        void addInputConstraints(
                size_t num_states,
                size_t num_inputs,
                Eigen::SparseMatrix<double>& constraint_matrix,
                Eigen::VectorXd& lower_bound,
                Eigen::VectorXd& upper_bound,
                std::vector<Eigen::Triplet<double>>& triplets) const;

        /**
         * @brief 添加状态约束到约束矩阵和边界向量
         * @param num_states 状态变量数量
         * @param constraint_matrix 约束矩阵（稀疏矩阵）
         * @param lower_bound 下界向量
         * @param upper_bound 上界向量
         * @param triplets 三元组列表（用于构建稀疏矩阵）
         */
        void addStateConstraints(
                size_t num_states,
                Eigen::SparseMatrix<double>& constraint_matrix,
                Eigen::VectorXd& lower_bound,
                Eigen::VectorXd& upper_bound,
                std::vector<Eigen::Triplet<double>>& triplets) const;

        /**
         * @brief 添加输入速率约束到约束矩阵和边界向量
         * @param num_states 状态变量数量
         * @param num_inputs 输入变量数量
         * @param constraint_matrix 约束矩阵（稀疏矩阵）
         * @param lower_bound 下界向量
         * @param upper_bound 上界向量
         * @param triplets 三元组列表（用于构建稀疏矩阵）
         */
        void addInputRateConstraints(
                size_t num_states,
                size_t num_inputs,
                Eigen::SparseMatrix<double>& constraint_matrix,
                Eigen::VectorXd& lower_bound,
                Eigen::VectorXd& upper_bound,
                std::vector<Eigen::Triplet<double>>& triplets) const;

        /**
         * @brief 添加初始状态约束到约束矩阵和边界向量
         * @param initial_state 初始状态
         * @param constraint_matrix 约束矩阵（稀疏矩阵）
         * @param lower_bound 下界向量
         * @param upper_bound 上界向量
         * @param triplets 三元组列表（用于构建稀疏矩阵）
         */
        void addInitialStateConstraints(
                const State& initial_state,
                Eigen::SparseMatrix<double>& constraint_matrix,
                Eigen::VectorXd& lower_bound,
                Eigen::VectorXd& upper_bound,
                std::vector<Eigen::Triplet<double>>& triplets) const;
    };

}  // namespace mpc