import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get the package share directory
    pkg_dir = get_package_share_directory('mpc_osqp_ros2')

    # Parameters file
    params_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    # MPC node
    mpc_node = Node(
        package='mpc_osqp_ros2',
        executable='mpc_node',
        name='mpc_controller',
        output='screen',
        parameters=[params_file],
        remappings=[
            ('odom', '/odometry/filtered'),
            ('plan', '/global_plan'),
            ('cmd_vel', '/cmd_vel'),
            ('predicted_path', '/mpc/predicted_path')
        ]
    )

    return LaunchDescription([
        mpc_node
    ])