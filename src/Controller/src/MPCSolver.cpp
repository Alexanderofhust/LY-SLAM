#include "mpc_osqp_ros2/MPCSolver.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace mpc {

    using namespace std::chrono;

    MPCSolver::MPCSolver() {
        // 初始化求解器
        solver_ = std::make_unique<OsqpEigen::Solver>();
    }

    MPCSolver::~MPCSolver() {
        // 清理资源
        if (solver_ && is_solver_setup_) {
            solver_->clearSolver();
        }
    }

    bool MPCSolver::setup(const MPCConfig& config) {
        config_ = config;

        // 验证配置参数
        if (config_.prediction_horizon == 0 || config_.control_horizon == 0) {
            solver_status_ = "Invalid prediction or control horizon";
            return false;
        }

        if (config_.time_step <= 0) {
            solver_status_ = "Invalid time step";
            return false;
        }

        // 计算问题维度
        const size_t state_dim = 3;  // x, y, theta
        const size_t input_dim = 2;  // v, omega

        const size_t num_states = config_.prediction_horizon * state_dim;
        const size_t num_inputs = config_.control_horizon * input_dim;
        const size_t num_variables = num_states + num_inputs;

        // 预分配矩阵内存
        hessian_.resize(num_variables, num_variables);
        constraint_matrix_.resize(num_variables, num_variables);
        gradient_.resize(num_variables);
        lower_bound_.resize(num_variables);
        upper_bound_.resize(num_variables);

        // 初始化系统矩阵序列
        A_matrices_.resize(config_.prediction_horizon);
        B_matrices_.resize(config_.prediction_horizon);

        // 标记为已初始化
        is_initialized_ = true;
        solver_status_ = "Initialized";

        return true;
    }

    bool MPCSolver::solve(const State& current_state,
                          const ReferenceTrajectory& reference,
                          ControlInput& optimal_input,
                          std::vector<State>* predicted_states) {
        // 检查是否已初始化
        if (!is_initialized_) {
            solver_status_ = "Solver not initialized";
            return false;
        }

        // 检查参考轨迹长度
        if (reference.states.size() < config_.prediction_horizon) {
            solver_status_ = "Reference trajectory too short";
            return false;
        }

        // 记录求解开始时间
        auto start_time = high_resolution_clock::now();

        try {
            // 构建QP问题
            constructHessian();
            constructGradient(current_state, reference);
            constructConstraintMatrix();
            constructConstraintBounds(current_state);

            // 设置求解器数据
            if (!is_solver_setup_) {
                // 初次设置求解器
                solver_->settings()->setVerbosity(false);
                solver_->settings()->setWarmStart(true);
                solver_->data()->setNumberOfVariables(hessian_.rows());
                solver_->data()->setNumberOfConstraints(constraint_matrix_.rows());

                if (!solver_->data()->setHessianMatrix(hessian_)) {
                    solver_status_ = "Failed to set Hessian matrix";
                    return false;
                }

                if (!solver_->data()->setGradient(gradient_)) {
                    solver_status_ = "Failed to set gradient vector";
                    return false;
                }

                if (!solver_->data()->setLinearConstraintsMatrix(constraint_matrix_)) {
                    solver_status_ = "Failed to set constraint matrix";
                    return false;
                }

                if (!solver_->data()->setLowerBound(lower_bound_)) {
                    solver_status_ = "Failed to set lower bound";
                    return false;
                }

                if (!solver_->data()->setUpperBound(upper_bound_)) {
                    solver_status_ = "Failed to set upper bound";
                    return false;
                }

                // 初始化求解器
                if (!solver_->initSolver()) {
                    solver_status_ = "Failed to initialize solver";
                    return false;
                }

                is_solver_setup_ = true;
            } else {
                // 更新求解器数据
                if (!solver_->updateHessianMatrix(hessian_)) {
                    solver_status_ = "Failed to update Hessian matrix";
                    return false;
                }

                if (!solver_->updateGradient(gradient_)) {
                    solver_status_ = "Failed to update gradient vector";
                    return false;
                }

                if (!solver_->updateLinearConstraintsMatrix(constraint_matrix_)) {
                    solver_status_ = "Failed to update constraint matrix";
                    return false;
                }

                if (!solver_->updateBounds(lower_bound_, upper_bound_)) {
                    solver_status_ = "Failed to update bounds";
                    return false;
                }
            }

            // 求解QP问题
            if (solver_->solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
                solver_status_ = "Failed to solve QP problem";
                return false;
            }

            // 获取解
            Eigen::VectorXd solution = solver_->getSolution();

            // 从解中提取控制输入和预测状态
            extractSolution(solution, optimal_input, predicted_states);

            // 记录求解时间
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            total_solve_time_ += duration.count() / 1000.0;  // 转换为毫秒
            solve_count_++;

            solver_status_ = "Solved successfully";
            return true;

        } catch (const std::exception& e) {
            solver_status_ = std::string("Exception during solve: ") + e.what();
            return false;
        }
    }

    void MPCSolver::updateModel(const SystemModel& model) {
        model_ = model;
    }

    void MPCSolver::updateConstraints(const Constraints& constraints) {
        constraints_ = constraints;
    }

    double MPCSolver::getAverageSolveTime() const {
        if (solve_count_ == 0) {
            return 0.0;
        }
        return total_solve_time_ / solve_count_;
    }

    void MPCSolver::resetSolveTimeStats() {
        total_solve_time_ = 0.0;
        solve_count_ = 0;
    }

    std::string MPCSolver::getSolverStatus() const {
        return solver_status_;
    }

    void MPCSolver::constructHessian() {
        const size_t state_dim = 3;
        const size_t input_dim = 2;
        const size_t num_states = config_.prediction_horizon * state_dim;
        const size_t num_inputs = config_.control_horizon * input_dim;
        const size_t num_variables = num_states + num_inputs;

        // 重置Hessian矩阵
        hessian_.setZero();
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(num_states + num_inputs);

        // 添加状态权重
        for (size_t i = 0; i < config_.prediction_horizon; ++i) {
            for (size_t j = 0; j < state_dim; ++j) {
                size_t idx = i * state_dim + j;
                triplets.emplace_back(idx, idx, config_.state_weight(j, j));
            }
        }

        // 添加输入权重
        for (size_t i = 0; i < config_.control_horizon; ++i) {
            for (size_t j = 0; j < input_dim; ++j) {
                size_t idx = num_states + i * input_dim + j;
                triplets.emplace_back(idx, idx, config_.input_weight(j, j));
            }
        }

        // 添加输入变化率权重（如果启用）
        if (config_.input_rate_weight.norm() > 0) {
            for (size_t i = 1; i < config_.control_horizon; ++i) {
                for (size_t j = 0; j < input_dim; ++j) {
                    size_t idx1 = num_states + (i - 1) * input_dim + j;
                    size_t idx2 = num_states + i * input_dim + j;

                    // 对角线元素
                    triplets.emplace_back(idx1, idx1, config_.input_rate_weight(j, j));
                    triplets.emplace_back(idx2, idx2, config_.input_rate_weight(j, j));

                    // 非对角线元素
                    triplets.emplace_back(idx1, idx2, -config_.input_rate_weight(j, j));
                    triplets.emplace_back(idx2, idx1, -config_.input_rate_weight(j, j));
                }
            }
        }

        // 设置Hessian矩阵
        hessian_.setFromTriplets(triplets.begin(), triplets.end());
    }

    void MPCSolver::constructGradient(const State& current_state,
                                      const ReferenceTrajectory& reference) {
        const size_t state_dim = 3;
        const size_t input_dim = 2;
        const size_t num_states = config_.prediction_horizon * state_dim;
        const size_t num_inputs = config_.control_horizon * input_dim;
        const size_t num_variables = num_states + num_inputs;

        // 重置梯度向量
        gradient_.setZero();

        // 计算状态误差的梯度部分
        for (size_t i = 0; i < config_.prediction_horizon; ++i) {
            Eigen::Vector3d state_error;
            if (i == 0) {
                state_error = current_state - reference.states[i];
            } else {
                // 这里使用预测的状态误差，实际实现中需要更精确的计算
                state_error = -reference.states[i];
            }

            for (size_t j = 0; j < state_dim; ++j) {
                gradient_(i * state_dim + j) = -2.0 * config_.state_weight(j, j) * state_error(j);
            }
        }

        // 计算输入误差的梯度部分
        for (size_t i = 0; i < config_.control_horizon; ++i) {
            Eigen::Vector2d input_error = -reference.inputs[i];

            for (size_t j = 0; j < input_dim; ++j) {
                gradient_(num_states + i * input_dim + j) = -2.0 * config_.input_weight(j, j) * input_error(j);
            }
        }
    }

    void MPCSolver::constructConstraintMatrix() {
        const size_t state_dim = 3;
        const size_t input_dim = 2;
        const size_t num_states = config_.prediction_horizon * state_dim;
        const size_t num_inputs = config_.control_horizon * input_dim;
        const size_t num_variables = num_states + num_inputs;

        // 重置约束矩阵
        constraint_matrix_.setZero();
        std::vector<Eigen::Triplet<double>> triplets;

        // 构建系统动力学约束
        // 这部分需要根据具体的系统模型实现
        // 这里是一个简化的示例实现

        // 添加单位矩阵部分（状态变量）
        for (size_t i = 0; i < num_states; ++i) {
            triplets.emplace_back(i, i, 1.0);
        }

        // 添加系统矩阵部分
        // 这部分需要根据系统动力学模型计算A和B矩阵
        // 这里使用简化的实现

        // 设置约束矩阵
        constraint_matrix_.setFromTriplets(triplets.begin(), triplets.end());
    }

    void MPCSolver::constructConstraintBounds(const State& current_state) {
        const size_t state_dim = 3;
        const size_t input_dim = 2;
        const size_t num_states = config_.prediction_horizon * state_dim;
        const size_t num_inputs = config_.control_horizon * input_dim;
        const size_t num_variables = num_states + num_inputs;

        // 使用Constraints类的统一方法构建约束
        constraints_.constructConstraints(
                config_.prediction_horizon,
                config_.control_horizon,
                state_dim,
                input_dim,
                constraint_matrix_,
                lower_bound_,
                upper_bound_,
                current_state
        );

        // 如果约束矩阵为空，则使用默认的单位矩阵约束
        if (constraint_matrix_.nonZeros() == 0) {
            RCLCPP_WARN(rclcpp::get_logger("mpc_solver"),
                        "Constraint matrix is empty, using identity constraints");

            // 重置约束矩阵和边界向量
            constraint_matrix_.resize(num_variables, num_variables);
            lower_bound_.resize(num_variables);
            upper_bound_.resize(num_variables);

            // 使用三元组列表构建稀疏矩阵
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(num_variables);

            // 添加单位矩阵约束
            for (size_t i = 0; i < num_variables; ++i) {
                triplets.emplace_back(i, i, 1.0);

                // 设置默认边界
                if (i < num_states) {
                    // 状态变量约束
                    size_t state_idx = i % state_dim;
                    if (constraints_.hasStateConstraints()) {
                        lower_bound_(i) = constraints_.getMinState()(state_idx);
                        upper_bound_(i) = constraints_.getMaxState()(state_idx);
                    } else {
                        // 无状态约束时使用极大值
                        lower_bound_(i) = -std::numeric_limits<double>::max();
                        upper_bound_(i) = std::numeric_limits<double>::max();
                    }
                } else {
                    // 输入变量约束
                    size_t input_idx = (i - num_states) % input_dim;
                    if (constraints_.hasInputConstraints()) {
                        lower_bound_(i) = constraints_.getMinInput()(input_idx);
                        upper_bound_(i) = constraints_.getMaxInput()(input_idx);
                    } else {
                        // 无输入约束时使用极大值
                        lower_bound_(i) = -std::numeric_limits<double>::max();
                        upper_bound_(i) = std::numeric_limits<double>::max();
                    }
                }
            }

            // 设置初始状态约束
            for (size_t i = 0; i < state_dim; ++i) {
                lower_bound_(i) = current_state(i);
                upper_bound_(i) = current_state(i);
            }

            constraint_matrix_.setFromTriplets(triplets.begin(), triplets.end());
        }

        // 验证约束矩阵和边界向量的维度是否一致
        if (constraint_matrix_.rows() != lower_bound_.size() ||
            constraint_matrix_.rows() != upper_bound_.size()) {
            RCLCPP_ERROR(rclcpp::get_logger("mpc_solver"),
                         "Constraint matrix and bounds dimension mismatch: %zu vs %zu vs %zu",
                         constraint_matrix_.rows(), lower_bound_.size(), upper_bound_.size());
            throw std::runtime_error("Constraint matrix and bounds dimension mismatch");
        }

        // 验证约束边界是否合理（下界 <= 上界）
        for (int i = 0; i < lower_bound_.size(); ++i) {
            if (lower_bound_(i) > upper_bound_(i)) {
                RCLCPP_ERROR(rclcpp::get_logger("mpc_solver"),
                             "Invalid constraint bounds at index %d: lower=%f, upper=%f",
                             i, lower_bound_(i), upper_bound_(i));
                throw std::runtime_error("Invalid constraint bounds");
            }
        }

        RCLCPP_DEBUG(rclcpp::get_logger("mpc_solver"),
                     "Constructed constraint matrix with %zu rows and %zu non-zero elements",
                     constraint_matrix_.rows(), constraint_matrix_.nonZeros());
    }

    void MPCSolver::extractSolution(const Eigen::VectorXd& solution,
                                    ControlInput& optimal_input,
                                    std::vector<State>* predicted_states) {
        const size_t state_dim = 3;
        const size_t input_dim = 2;
        const size_t num_states = config_.prediction_horizon * state_dim;

        // 提取第一个控制输入作为最优控制
        optimal_input(0) = solution(num_states);      // v
        optimal_input(1) = solution(num_states + 1);  // ω

        // 提取预测状态序列（如果请求）
        if (predicted_states != nullptr) {
            predicted_states->clear();
            predicted_states->reserve(config_.prediction_horizon);

            for (size_t i = 0; i < config_.prediction_horizon; ++i) {
                State state;
                state(0) = solution(i * state_dim);      // x
                state(1) = solution(i * state_dim + 1);  // y
                state(2) = solution(i * state_dim + 2);  // θ
                predicted_states->push_back(state);
            }
        }
    }

}  // namespace mpc