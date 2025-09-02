cmake_minimum_required(VERSION 3.8)
project(rog_map)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set C++ standard to 14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -Wall -g -fPIC")
add_compile_options(-DQT_NO_VERSION_TAGGING)
add_compile_options(-Werror=unused-variable)
add_compile_options(-Werror=return-type)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

add_definitions(-DORIGIN_AT_CORNER)
#add_definitions(-DORIGIN_AT_CENTER)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(builtin_interfaces REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

# Add dynamic reconfigure equivalent for ROS2
# Note: ROS2 uses parameters instead of dynamic_reconfigure

# Export information to downstream packages
set(THIS_PACKAGE_INCLUDE_DEPENDS
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  pcl_ros
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
  message_filters
  builtin_interfaces
)

include_directories(
        SYSTEM
        include
        ../../../install/rog_map/include
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
)

set(THIRD_PARTY
        ${PCL_LIBRARIES}
        -ldw
        )

# Add libraries
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h include/*.hpp include/*.cpp)
add_library(rog_map SHARED ${srcs})
target_include_directories(rog_map PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(rog_map ${THIS_PACKAGE_INCLUDE_DEPENDS})

# Create executables
# Note: You'll need to create specific executables based on your main functions
# add_executable(rog_map_node src/rog_map_node.cpp)
# ament_target_dependencies(rog_map_node ${THIS_PACKAGE_INCLUDE_DEPENDS})
# target_link_libraries(rog_map_node rog_map ${THIRD_PARTY})

# Install targets
install(TARGETS rog_map
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Export dependencies
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(rog_map)

ament_package()