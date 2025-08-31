cmake_minimum_required(VERSION 3.16)
project(mpc_osqp_ros2)

# Default to C++20 (ROS2 Humble supports C++20)
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

# ROS2 setup
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# External dependencies
find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)

# Include directories
include_directories(
        include
        ${Eigen3_INCLUDE_DIRS}
        ${OsqpEigen_INCLUDE_DIRS}
)

# MPC library
add_library(mpc_controller
        src/MPCSolver.cpp
        src/SystemModel.cpp
        src/Constraints.cpp
)
ament_target_dependencies(mpc_controller
        rclcpp
        geometry_msgs
        nav_msgs
        tf2
        tf2_ros
        tf2_geometry_msgs
        Eigen3
        OsqpEigen
)

# ROS2 node
add_executable(mpc_node
        src/MPCNode.cpp
)
ament_target_dependencies(mpc_node
        rclcpp
        geometry_msgs
        nav_msgs
        tf2
        tf2_ros
        tf2_geometry_msgs
        Eigen3
        OsqpEigen
)
target_link_libraries(mpc_node mpc_controller)

# Install targets
install(TARGETS
        mpc_controller
        mpc_node
        DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
        include/
        DESTINATION include
)

install(DIRECTORY
        launch/
        config/
        DESTINATION share/${PROJECT_NAME}
)

# Tests
if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)

    ament_add_gtest(test_mpc_solver test/test_mpc_solver.cpp)
    target_link_libraries(test_mpc_solver mpc_controller)
    ament_target_dependencies(test_mpc_solver rclcpp)

    ament_add_gtest(test_system_model test/test_system_model.cpp)
    target_link_libraries(test_system_model mpc_controller)
    ament_target_dependencies(test_system_model rclcpp)

    ament_add_gtest(test_constraints test/test_constraints.cpp)
    target_link_libraries(test_constraints mpc_controller)
    ament_target_dependencies(test_constraints rclcpp)

    ament_add_gtest(test_mpc_node test/test_mpc_node.cpp)
    target_link_libraries(test_mpc_node mpc_controller)
    ament_target_dependencies(test_mpc_node
            rclcpp
            geometry_msgs
            nav_msgs
            tf2
            tf2_ros
            tf2_geometry_msgs
    )
endif()

# Export dependencies
ament_export_include_directories(include)
ament_export_libraries(mpc_controller)
ament_export_dependencies(
        rclcpp
        geometry_msgs
        nav_msgs
        tf2
        tf2_ros
        tf2_geometry_msgs
        Eigen3
        OsqpEigen
)

ament_package()
