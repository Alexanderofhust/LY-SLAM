#include "mpc_osqp_ros2/Constraints.hpp"
#include <stdexcept>
#include <iostream>

namespace mpc {

    Constraints::Constraints() {
        // 初始化默认约束值
        min_input_.setZero();
        max_input_.setZero();
        min_state_.setZero();
        max_state_.setZero();
        min_input_rate_.setZero();
        max_input_rate_.setZero();
    }

    Constraints::~Constraints() {
        // 清理资源（如果需要）
    }

    void Constraints::setInputConstraints(const ControlInput& min_input, const ControlInput& max_input) {
        // 验证输入约束的合理性
        if (min_input.size() != max_input.size()) {
            throw std::invalid_argument("Input constraint vectors must have the same size");
        }

        for (int i = 0; i < min_input.size(); ++i) {
            if (min_input(i) > max_input(i)) {
                throw std::invalid_argument("Minimum input cannot be greater than maximum input");
            }
        }

        min_input_ = min_input;
        max_input_ = max_input;
        has_input_constraints_ = true;
    }

    void Constraints::setStateConstraints(const State& min_state, const State& max_state) {
        // 验证状态约束的合理性
        if (min_state.size() != max_state.size()) {
            throw std::invalid_argument("State constraint vectors must have the same size");
        }

        for (int i = 0; i < min_state.size(); ++i) {
            if (min_state(i) > max_state(i)) {
                throw std::invalid_argument("Minimum state cannot be greater than maximum state");
            }
        }

        min_state_ = min_state;
        max_state_ = max_state;
        has_state_constraints_ = true;
    }

    void Constraints::setInputRateConstraints(const ControlInput& min_rate, const ControlInput& max_rate) {
        // 验证输入速率约束的合理性
        if (min_rate.size() != max_rate.size()) {
            throw std::invalid_argument("Input rate constraint vectors must have the same size");
        }

        for (int i = 0; i < min_rate.size(); ++i) {
            if (min_rate(i) > max_rate(i)) {
                throw std::invalid_argument("Minimum input rate cannot be greater than maximum input rate");
            }
        }

        min_input_rate_ = min_rate;
        max_input_rate_ = max_rate;
        has_input_rate_constraints_ = true;
    }

    ControlInput Constraints::getMinInput() const {
        return min_input_;
    }

    ControlInput Constraints::getMaxInput() const {
        return max_input_;
    }

    State Constraints::getMinState() const {
        return min_state_;
    }

    State Constraints::getMaxState() const {
        return max_state_;
    }

    ControlInput& Constraints::getMinInputRate() const {
        return min_input_rate_;
    }

    const ControlInput& Constraints::getMaxInputRate() const {
        return max_input_rate_;
    }

    bool Constraints::hasInputConstraints() const {
        return has_input_constraints_;
    }

    bool Constraints::hasStateConstraints() const {
        return has_state_constraints_;
    }

    bool Constraints::hasInputRateConstraints() const {
        return has_input_rate_constraints_;
    }

    void Constraints::constructConstraints(
            size_t prediction_horizon,
            size_t control_horizon,
            size_t state_dim,
            size_t input_dim,
            Eigen::SparseMatrix<double>& constraint_matrix,
            Eigen::VectorXd& lower_bound,
            Eigen::VectorXd& upper_bound,
            const State& initial_state) const {

        // 计算变量总数
        const size_t num_states = prediction_horizon * state_dim;
        const size_t num_inputs = control_horizon * input_dim;
        const size_t num_variables = num_states + num_inputs;

        // 计算约束总数
        size_t num_constraints = 0;

        // 初始状态约束
        num_constraints += state_dim;

        // 状态约束
        if (has_state_constraints_) {
            num_constraints += num_states;
        }

        // 输入约束
        if (has_input_constraints_) {
            num_constraints += num_inputs;
        }

        // 输入速率约束
        if (has_input_rate_constraints_) {
            num_constraints += (control_horizon - 1) * input_dim;
        }

        // 重置约束矩阵和边界向量
        constraint_matrix.resize(num_constraints, num_variables);
        lower_bound.resize(num_constraints);
        upper_bound.resize(num_constraints);

        // 使用三元组列表构建稀疏矩阵
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(num_constraints);  // 保守估计

        // 重置边界向量
        lower_bound.setZero();
        upper_bound.setZero();

        // 添加初始状态约束
        addInitialStateConstraints(initial_state, constraint_matrix, lower_bound, upper_bound, triplets);

        // 添加状态约束
        if (has_state_constraints_) {
            addStateConstraints(num_states, constraint_matrix, lower_bound, upper_bound, triplets);
        }

        // 添加输入约束
        if (has_input_constraints_) {
            addInputConstraints(num_states, num_inputs, constraint_matrix, lower_bound, upper_bound, triplets);
        }

        // 添加输入速率约束
        if (has_input_rate_constraints_) {
            addInputRateConstraints(num_states, num_inputs, constraint_matrix, lower_bound, upper_bound, triplets);
        }

        // 设置约束矩阵
        constraint_matrix.setFromTriplets(triplets.begin(), triplets.end());
    }

    void Constraints::reset() {
        min_input_.setZero();
        max_input_.setZero();
        min_state_.setZero();
        max_state_.setZero();
        min_input_rate_.setZero();
        max_input_rate_.setZero();

        has_input_constraints_ = false;
        has_state_constraints_ = false;
        has_input_rate_constraints_ = false;
    }

    void Constraints::addInputConstraints(
            size_t num_states,
            size_t num_inputs,
            Eigen::SparseMatrix<double>& constraint_matrix,
            Eigen::VectorXd& lower_bound,
            Eigen::VectorXd& upper_bound,
            std::vector<Eigen::Triplet<double>>& triplets) const {

        // 添加输入约束：u_min ≤ u ≤ u_max
        for (size_t i = 0; i < num_inputs; ++i) {
            size_t constraint_idx = triplets.size() / 2;  // 估计约束索引
            size_t variable_idx = num_states + i;

            // 添加单位矩阵元素
            triplets.emplace_back(constraint_idx, variable_idx, 1.0);

            // 设置边界
            lower_bound(constraint_idx) = min_input_(i % min_input_.size());
            upper_bound(constraint_idx) = max_input_(i % max_input_.size());
        }
    }

    void Constraints::addStateConstraints(
            size_t num_states,
            Eigen::SparseMatrix<double>& constraint_matrix,
            Eigen::VectorXd& lower_bound,
            Eigen::VectorXd& upper_bound,
            std::vector<Eigen::Triplet<double>>& triplets) const {

        // 添加状态约束：x_min ≤ x ≤ x_max
        for (size_t i = 0; i < num_states; ++i) {
            size_t constraint_idx = triplets.size() / 2;  // 估计约束索引

            // 添加单位矩阵元素
            triplets.emplace_back(constraint_idx, i, 1.0);

            // 设置边界
            lower_bound(constraint_idx) = min_state_(i % min_state_.size());
            upper_bound(constraint_idx) = max_state_(i % max_state_.size());
        }
    }

    void Constraints::addInputRateConstraints(
            size_t num_states,
            size_t num_inputs,
            Eigen::SparseMatrix<double>& constraint_matrix,
            Eigen::VectorXd& lower_bound,
            Eigen::VectorXd& upper_bound,
            std::vector<Eigen::Triplet<double>>& triplets) const {

        const size_t input_dim = min_input_rate_.size();
        const size_t control_horizon = num_inputs / input_dim;

        // 添加输入速率约束：Δu_min ≤ u_{k+1} - u_k ≤ Δu_max
        for (size_t i = 0; i < control_horizon - 1; ++i) {
            for (size_t j = 0; j < input_dim; ++j) {
                size_t constraint_idx = triplets.size() / 2;  // 估计约束索引
                size_t var_idx1 = num_states + i * input_dim + j;
                size_t var_idx2 = num_states + (i + 1) * input_dim + j;

                // 添加差分约束：u_{k+1} - u_k
                triplets.emplace_back(constraint_idx, var_idx2, 1.0);
                triplets.emplace_back(constraint_idx, var_idx1, -1.0);

                // 设置边界
                lower_bound(constraint_idx) = min_input_rate_(j);
                upper_bound(constraint_idx) = max_input_rate_(j);
            }
        }
    }

    void Constraints::addInitialStateConstraints(
            const State& initial_state,
            Eigen::SparseMatrix<double>& constraint_matrix,
            Eigen::VectorXd& lower_bound,
            Eigen::VectorXd& upper_bound,
            std::vector<Eigen::Triplet<double>>& triplets) const {

        // 添加初始状态约束：x_0 = initial_state
        for (size_t i = 0; i < initial_state.size(); ++i) {
            // 添加单位矩阵元素
            triplets.emplace_back(i, i, 1.0);

            // 设置边界（等式约束）
            lower_bound(i) = initial_state(i);
            upper_bound(i) = initial_state(i);
        }
    }

}  // namespace mpc