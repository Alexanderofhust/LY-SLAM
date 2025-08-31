#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "mpc_osqp_ros2/Constraints.hpp"

class ConstraintsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置约束
        constraints_.setInputConstraints(
                mpc::ControlInput(-0.5, -1.0),
                mpc::ControlInput(0.5, 1.0)
        );
        constraints_.setStateConstraints(
                mpc::State(-10.0, -10.0, -3.14),
                mpc::State(10.0, 10.0, 3.14)
        );
        constraints_.setInputRateConstraints(
                mpc::ControlInput(-0.2, -0.5),
                mpc::ControlInput(0.2, 0.5)
        );
    }

    mpc::Constraints constraints_;
};

// 测试约束验证
TEST_F(ConstraintsTest, ConstraintValidation) {
// 测试无效约束（最小值大于最大值）
EXPECT_THROW(
        constraints_.setInputConstraints(
        mpc::ControlInput(0.5, 1.0), // 最小值
mpc::ControlInput(-0.5, -1.0) // 最大值（小于最小值）
),
std::invalid_argument
);
}

// 测试约束获取
TEST_F(ConstraintsTest, ConstraintGetters) {
mpc::ControlInput min_input = constraints_.getMinInput();
mpc::ControlInput max_input = constraints_.getMaxInput();

EXPECT_NEAR(min_input(0), -0.5, 1e-6);
EXPECT_NEAR(min_input(1), -1.0, 1e-6);
EXPECT_NEAR(max_input(0), 0.5, 1e-6);
EXPECT_NEAR(max_input(1), 1.0, 1e-6);

EXPECT_TRUE(constraints_.hasInputConstraints());
EXPECT_TRUE(constraints_.hasStateConstraints());
EXPECT_TRUE(constraints_.hasInputRateConstraints());
}

// 测试约束构建
TEST_F(ConstraintsTest, ConstraintConstruction) {
Eigen::SparseMatrix<double> constraint_matrix;
Eigen::VectorXd lower_bound, upper_bound;
mpc::State initial_state(0.0, 0.0, 0.0);

constraints_.constructConstraints(
10, // prediction_horizon
5,  // control_horizon
3,  // state_dim
2,  // input_dim
constraint_matrix,
lower_bound,
upper_bound,
initial_state
);

// 验证约束矩阵和边界维度
const size_t num_variables = 10 * 3 + 5 * 2; // 状态变量 + 输入变量
EXPECT_EQ(constraint_matrix.rows(), lower_bound.size());
EXPECT_EQ(constraint_matrix.rows(), upper_bound.size());
EXPECT_GT(constraint_matrix.rows(), 0);

// 验证初始状态约束
for (int i = 0; i < 3; ++i) {
EXPECT_NEAR(lower_bound(i), initial_state(i), 1e-6);
EXPECT_NEAR(upper_bound(i), initial_state(i), 1e-6);
}
}

// 测试约束重置
TEST_F(ConstraintsTest, ConstraintReset) {
constraints_.reset();

EXPECT_FALSE(constraints_.hasInputConstraints());
EXPECT_FALSE(constraints_.hasStateConstraints());
EXPECT_FALSE(constraints_.hasInputRateConstraints());
}

// 测试无约束情况
TEST_F(ConstraintsTest, NoConstraints) {
mpc::Constraints no_constraints;
Eigen::SparseMatrix<double> constraint_matrix;
Eigen::VectorXd lower_bound, upper_bound;
mpc::State initial_state(0.0, 0.0, 0.0);

no_constraints.constructConstraints(
10, 5, 3, 2,
constraint_matrix,
lower_bound,
upper_bound,
initial_state
);

// 即使没有约束，也应该构建初始状态约束
EXPECT_GT(constraint_matrix.rows(), 0);

// 验证初始状态约束
for (int i = 0; i < 3; ++i) {
EXPECT_NEAR(lower_bound(i), initial_state(i), 1e-6);
EXPECT_NEAR(upper_bound(i), initial_state(i), 1e-6);
}
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}