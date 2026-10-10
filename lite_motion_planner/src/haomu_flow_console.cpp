#include "lite_motion_planner/robot_model.hpp"

#include <lite_motion_msgs/srv/add_scene_object.hpp>
#include <lite_motion_msgs/srv/attach_scene_object.hpp>
#include <lite_motion_msgs/srv/plan_arm_motion.hpp>
#include <lite_motion_msgs/srv/remove_scene_object.hpp>
#include <lite_motion_msgs/srv/set_scene_collision_allowance.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rcutils/logging.h>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <Eigen/Geometry>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace lite_motion_planner {
namespace {

geometry_msgs::msg::Quaternion quaternionMsgFromRpy(double roll, double pitch, double yaw) {
  const Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
  const Eigen::Quaterniond q = rz * ry * rx;

  geometry_msgs::msg::Quaternion out;
  out.x = q.x();
  out.y = q.y();
  out.z = q.z();
  out.w = q.w();
  return out;
}

Eigen::Quaterniond quaternionEigenFromMsg(const geometry_msgs::msg::Quaternion& q_msg) {
  Eigen::Quaterniond q(q_msg.w, q_msg.x, q_msg.y, q_msg.z);
  if (q.norm() < 1e-12) {
    q = Eigen::Quaterniond::Identity();
  }
  q.normalize();
  return q;
}

geometry_msgs::msg::Pose poseMsgFromSE3(const pinocchio::SE3& tf) {
  geometry_msgs::msg::Pose out;
  out.position.x = tf.translation().x();
  out.position.y = tf.translation().y();
  out.position.z = tf.translation().z();
  const Eigen::Quaterniond q(tf.rotation());
  out.orientation.x = q.x();
  out.orientation.y = q.y();
  out.orientation.z = q.z();
  out.orientation.w = q.w();
  return out;
}

pinocchio::SE3 poseMsgToSE3(const geometry_msgs::msg::Pose& pose) {
  return pinocchio::SE3(
    quaternionEigenFromMsg(pose.orientation).toRotationMatrix(),
    Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z));
}

double wrapToPi(double angle_rad) {
  return std::atan2(std::sin(angle_rad), std::cos(angle_rad));
}

double trajectoryDurationSec(const trajectory_msgs::msg::JointTrajectory& traj) {
  if (traj.points.empty()) {
    return 0.0;
  }
  const auto& last = traj.points.back().time_from_start;
  return static_cast<double>(last.sec) + static_cast<double>(last.nanosec) * 1e-9;
}

}  // namespace

class HaomuFlowConsoleNode : public rclcpp::Node {
public:
  HaomuFlowConsoleNode()
  : Node("haomu_flow_console") {
    const auto logger_level_result =
      rcutils_logging_set_logger_level(this->get_logger().get_name(), RCUTILS_LOG_SEVERITY_WARN);
    (void)logger_level_result;
    initializeConsoleInput();
    const auto urdf_path = this->declare_parameter<std::string>("urdf_path", "");
    const auto srdf_path = this->declare_parameter<std::string>("srdf_path", "");
    const auto mesh_package_dirs = this->declare_parameter<std::vector<std::string>>(
      "mesh_package_dirs", std::vector<std::string>());

    planning_frame_ = this->declare_parameter<std::string>("planning_frame", "base_link");
    left_arm_group_name_ = this->declare_parameter<std::string>("haomu_flow.left_arm_group", "left_arm");
    left_hand_group_name_ = this->declare_parameter<std::string>("haomu_flow.left_hand_group", "left_hand");
    left_tcp_frame_ = this->declare_parameter<std::string>("haomu_flow.left_tcp_frame", "left_tcp");
    left_ccb_frame_ = this->declare_parameter<std::string>("haomu_flow.left_ccb_frame", "left_ccb");
    right_arm_group_name_ = this->declare_parameter<std::string>("haomu_flow.right_arm_group", "right_arm");
    right_hand_group_name_ = this->declare_parameter<std::string>("haomu_flow.right_hand_group", "right_hand");
    right_tcp_frame_ = this->declare_parameter<std::string>("haomu_flow.right_tcp_frame", "right_tcp");
    right_ccb_frame_ = this->declare_parameter<std::string>("haomu_flow.right_ccb_frame", "right_ccb");
    right_arm_group_ = right_arm_group_name_;
    right_hand_group_ = right_hand_group_name_;
    tcp_frame_ = right_tcp_frame_;
    ccb_frame_ = right_ccb_frame_;

    e3_object_id_ = this->declare_parameter<std::string>("dhz4.e3_object_id", "E3");
    e3_mesh_resource_ = this->declare_parameter<std::string>("dhz4.e3_mesh_resource", "/home/zwt/E3.STL");
    zz_object_id_ = this->declare_parameter<std::string>("dhz4.zz_object_id", "zz");
    zz_mesh_resource_ = this->declare_parameter<std::string>("dhz4.zz_mesh_resource", "/home/zwt/zz.STL");
    dhz_object_id_ = this->declare_parameter<std::string>("dhz4.dhz_object_id", "dhz");
    dhz_mesh_resource_ = this->declare_parameter<std::string>("dhz4.dhz_mesh_resource", "/home/zwt/dhz.STL");
    mesh_scale_ = this->declare_parameter<std::vector<double>>(
      "dhz4.mesh_scale", std::vector<double>{0.001, 0.001, 0.001});

    d_test_xyz_ = declareVec3Param("dhz4.d_test_xyz", {0.5, -0.25, 0.0});
    d_test_rpy_ = declareVec3Param("dhz4.d_test_rpy", {1.57, 0.0, 0.0});
    p_test_xyz_ = declareVec3Param("haomu_flow.p_test_xyz", {0.5, -0.25, 0.0});
    p_test_rpy_ = declareVec3Param("haomu_flow.p_test_rpy", {0.0, 0.0, 0.0});
    catch_lift_distance_m_ = this->declare_parameter<double>("haomu_flow.catch_lift_distance_m", 0.11);
    default_recatch_side_ = this->declare_parameter<int>("haomu_flow.default_recatch_arm", 1);

    zc_f_xyz_ = declareVec3Param("dhz4.zc_f_xyz", {0.185, 0.0, 0.0725});
    zc_f_rpy_ = declareVec3Param("dhz4.zc_f_rpy", {0.0, 0.0, 0.0});
    zc_m_xyz_ = declareVec3Param("dhz4.zc_m_xyz", {0.085, 0.0, 0.0725});
    zc_m_rpy_ = declareVec3Param("dhz4.zc_m_rpy", {0.0, 0.0, 0.0});
    zc_l_xyz_ = declareVec3Param("dhz4.zc_l_xyz", {-0.0016, 0.050, 0.0725});
    zc_l_rpy_ = declareVec3Param("dhz4.zc_l_rpy", {0.0, 0.0, 0.0});
    zc_r_xyz_ = declareVec3Param("dhz4.zc_r_xyz", {-0.0016, -0.050, 0.0725});
    zc_r_rpy_ = declareVec3Param("dhz4.zc_r_rpy", {0.0, 0.0, 0.0});
    zc_target_name_ = this->declare_parameter<std::string>("dhz4.zc_target", "zc_f");
    if (zc_target_name_ != "zc_f" && zc_target_name_ != "zc_m" &&
        zc_target_name_ != "zc_l" && zc_target_name_ != "zc_r") {
      RCLCPP_WARN(this->get_logger(), "Unknown dhz4.zc_target [%s], falling back to zc_f", zc_target_name_.c_str());
      zc_target_name_ = "zc_f";
    }
    remove_unused_zc_e3_after_grasp_ = this->declare_parameter<bool>("dhz4.remove_unused_zc_e3_after_grasp", true);
    remove_zc_e3_objects_before_return_home_ = this->declare_parameter<bool>("dhz4.remove_zc_e3_objects_before_return_home", true);

    zc_use_position_focused_goal_ = this->declare_parameter<bool>("dhz4.zc_use_position_focused_goal", true);
    zc_orientation_weight_x_ = std::max(0.0, this->declare_parameter<double>("dhz4.zc_orientation_weight_x", 0.015));
    zc_orientation_weight_y_ = std::max(0.0, this->declare_parameter<double>("dhz4.zc_orientation_weight_y", 0.015));
    zc_orientation_weight_z_ = std::max(0.0, this->declare_parameter<double>("dhz4.zc_orientation_weight_z", 0.0));
    zc_ik_max_iterations_ = std::max<int>(50, this->declare_parameter<int>("dhz4.zc_ik_max_iterations", 260));
    zc_ik_pos_tolerance_ = std::max(1e-5, this->declare_parameter<double>("dhz4.zc_ik_pos_tolerance", 0.003));
    zc_ik_rot_tolerance_ = std::max(1e-5, this->declare_parameter<double>("dhz4.zc_ik_rot_tolerance", 0.15));
    zc_ik_damping_ = std::max(1e-12, this->declare_parameter<double>("dhz4.zc_ik_damping", 1.0e-4));
    zc_ik_alpha_ = std::max(1e-3, this->declare_parameter<double>("dhz4.zc_ik_alpha", 0.35));
    zc_ik_max_step_norm_ = std::max(1e-3, this->declare_parameter<double>("dhz4.zc_ik_max_step_norm", 0.50));
    zc_ik_max_seed_count_ = std::max<int>(1, this->declare_parameter<int>("dhz4.zc_ik_max_seed_count", 14));
    zc_planning_timeout_ = std::max(0.05, this->declare_parameter<double>("dhz4.zc_planning_timeout", 1.2));
    zc_path_check_step_ = std::max(0.005, this->declare_parameter<double>("dhz4.zc_path_check_step", 0.07));
    zc_ompl_max_plan_attempts_ = std::max<int>(1, this->declare_parameter<int>("dhz4.zc_ompl_max_plan_attempts", 1));
    zc_max_ik_goal_candidates_ = std::max<int>(1, this->declare_parameter<int>("dhz4.zc_max_ik_goal_candidates", 4));
    zc_relaxed_allowed_penetration_m_ = std::max(0.0, this->declare_parameter<double>("dhz4.zc_relaxed_scene_collision_allowed_penetration", 0.005));
    right_zc_relaxed_links_ = this->declare_parameter<std::vector<std::string>>(
      "dhz4.zc_relaxed_scene_collision_links",
      std::vector<std::string>{"cun_1_link", "cun_2_link", "cun_3_link", "cun_4_link", "right_tcp", "right_ccb", "arm_right_J7_link", "zhua_right_1_link", "zhua_right_2_link"});
    left_zc_relaxed_links_ = this->declare_parameter<std::vector<std::string>>(
      "haomu_flow.left_zc_relaxed_scene_collision_links",
      std::vector<std::string>{"cun_1_link", "cun_2_link", "cun_3_link", "cun_4_link", "left_tcp", "left_ccb", "arm_left_J7_link", "zhua_left_1_link", "zhua_left_2_link"});

    home_planning_timeout_ = std::max(0.05, this->declare_parameter<double>("dhz4.home_planning_timeout", 2.0));
    home_path_check_step_ = std::max(0.005, this->declare_parameter<double>("dhz4.home_path_check_step", 0.10));
    home_ompl_max_plan_attempts_ = std::max<int>(1, this->declare_parameter<int>("dhz4.home_ompl_max_plan_attempts", 3));
    return_home_relaxed_allowed_penetration_m_ = std::max(0.0, this->declare_parameter<double>("dhz4.return_home_relaxed_scene_collision_allowed_penetration", 0.005));
    return_home_relaxed_object_ids_ = this->declare_parameter<std::vector<std::string>>(
      "dhz4.return_home_relaxed_scene_collision_object_ids", std::vector<std::string>{dhz_object_id_});
    right_return_home_relaxed_links_ = this->declare_parameter<std::vector<std::string>>(
      "dhz4.return_home_relaxed_scene_collision_links",
      std::vector<std::string>{"right_tcp", "right_ccb", "arm_right_J7_link", "zhua_right_1_link", "zhua_right_2_link"});
    left_return_home_relaxed_links_ = this->declare_parameter<std::vector<std::string>>(
      "haomu_flow.left_return_home_relaxed_scene_collision_links",
      std::vector<std::string>{"left_tcp", "left_ccb", "arm_left_J7_link", "zhua_left_1_link", "zhua_left_2_link"});

    ccb_approach_distance_m_ = this->declare_parameter<double>("dhz4.ccb_approach_distance_m", 0.100);
    zz_y_push_distance_m_ = this->declare_parameter<double>("dhz4.zz_y_push_distance_m", 0.100);
    zz_attach_point_xyz_in_zz_ = declareVec3Param("dhz4.zz_attach_point_xyz_in_zz", {0.0, 0.0, -0.0275});
    zz_attach_point_xyz_in_e3_ = declareVec3Param("dhz4.zz_attach_point_xyz_in_e3", {0.0, 0.0, 0.0});
    zz_attach_rpy_in_e3_ = declareVec3Param("dhz4.zz_attach_rpy_in_e3", {0.0, 0.0, 0.0});
    arc7_center_xyz_dhz_ = declareVec3Param("dhz4.arc7_center_xyz_dhz", {0.0, 0.208, -0.169});
    arc7_roll_angle_rad_ = this->declare_parameter<double>("dhz4.arc7_roll_angle_rad", -1.5707963267948966);
    arc8_center_xyz_dhz_ = declareVec3Param("dhz4.arc8_center_xyz_dhz", {0.0, 0.223, -0.064});
    arc8_yaw_angle_abs_rad_ = std::abs(this->declare_parameter<double>("dhz4.arc8_yaw_angle_abs_rad", 1.5707963267948966));
    r_or_l_ = this->declare_parameter<int>("dhz4.r_or_l", 0);
    cartesian_max_step_m_ = std::max(1e-4, this->declare_parameter<double>("dhz4.cartesian_max_step_m", 0.08));
    cartesian_use_sdls_ = this->declare_parameter<bool>("dhz4.cartesian_use_sdls", true);
    cartesian_nullspace_gain_ = std::max(
      0.0,
      this->declare_parameter<double>("dhz4.cartesian_nullspace_gain", 0.20));
    cartesian_singular_threshold_ = std::max(
      1e-6,
      this->declare_parameter<double>("dhz4.cartesian_singular_threshold", 0.04));
    cartesian_branch_jump_max_rad_ = std::max(
      1e-6,
      this->declare_parameter<double>("dhz4.cartesian_branch_jump_max_rad", 0.75));
    cartesian_branch_jump_norm_max_ = std::max(
      1e-6,
      this->declare_parameter<double>("dhz4.cartesian_branch_jump_norm_max", 1.20));
    cartesian_waypoint_pos_tolerance_ = std::max(
      1e-6,
      this->declare_parameter<double>("dhz4.cartesian_waypoint_pos_tolerance", 0.0015));
    cartesian_waypoint_rot_tolerance_ = std::max(
      1e-6,
      this->declare_parameter<double>("dhz4.cartesian_waypoint_rot_tolerance", 0.020));
    arc_cartesian_max_step_m_ = std::max(1e-4, this->declare_parameter<double>("dhz4.arc_cartesian_max_step_m", 0.12));
    arc_min_step_count_ = std::max<int>(1, this->declare_parameter<int>("dhz4.arc_min_step_count", 2));
    // Exact waypoints are tried first. If a waypoint has no collision-free IK,
    // sample yaw about frozen dhz local +Z in parallel. Applies through arc7 only;
    // arc8 is exact-RPY only.
    arc8_cartesian_max_step_m_ = std::max(1e-4, this->declare_parameter<double>("dhz4.arc8_cartesian_max_step_m", 0.12));
    arc8_min_step_count_ = std::max<int>(1, this->declare_parameter<int>("dhz4.arc8_min_step_count", 3));
    yaw_sample_range_ = declareVec2Param("dhz4.yaw_sample_range", {0.79, 2.3});
    yaw_sample_count_ = std::max<int>(1, this->declare_parameter<int>("dhz4.yaw_sample_count", 6));
    arc8_relaxed_allowed_penetration_m_ = std::max(0.0, this->declare_parameter<double>("dhz4.arc8_relaxed_scene_collision_allowed_penetration", 0.010));
    arc8_relaxed_object_ids_ = this->declare_parameter<std::vector<std::string>>(
      "dhz4.arc8_relaxed_scene_collision_object_ids", std::vector<std::string>{dhz_object_id_});
    right_arc8_relaxed_links_ = this->declare_parameter<std::vector<std::string>>(
      "dhz4.arc8_relaxed_scene_collision_links", std::vector<std::string>{"right_tcp", "right_ccb", "arm_right_J7_link", "zhua_right_1_link", "zhua_right_2_link"});
    left_arc8_relaxed_links_ = this->declare_parameter<std::vector<std::string>>(
      "haomu_flow.left_arc8_relaxed_scene_collision_links",
      std::vector<std::string>{"left_tcp", "left_ccb", "arm_left_J7_link", "zhua_left_1_link", "zhua_left_2_link"});
    detach_zz_during_attached_chain_yaw_sweep_ = this->declare_parameter<bool>("dhz4.detach_zz_during_attached_chain_yaw_sweep", true);
    zz_global_ignore_links_ = this->declare_parameter<std::vector<std::string>>(
      "dhz4.zz_global_ignore_links", defaultRobotCollisionLinks());
    zz_global_ignore_allowed_penetration_m_ = this->declare_parameter<double>(
      "dhz4.zz_global_ignore_allowed_penetration", -1.0);
    e3_global_ignore_links_ = this->declare_parameter<std::vector<std::string>>(
      "dhz4.e3_global_ignore_links", defaultRobotCollisionLinks());
    e3_global_ignore_allowed_penetration_m_ = this->declare_parameter<double>(
      "dhz4.e3_global_ignore_allowed_penetration", -1.0);
    dhz_global_ignore_links_ = this->declare_parameter<std::vector<std::string>>(
      "dhz4.dhz_global_ignore_links", defaultRobotCollisionLinks());
    dhz_global_ignore_allowed_penetration_m_ = this->declare_parameter<double>(
      "dhz4.dhz_global_ignore_allowed_penetration", -1.0);
    post_action_wait_sec_ = this->declare_parameter<double>("dhz4.post_action_wait_sec", 0.05);
    service_wait_sec_ = this->declare_parameter<double>("dhz4.service_wait_sec", 30.0);
    response_wait_sec_ = this->declare_parameter<double>("dhz4.response_wait_sec", 30.0);
    sequence_loop_ = this->declare_parameter<bool>("dhz4.sequence_loop", false);
    loop_pause_sec_ = std::max(0.0, this->declare_parameter<double>("dhz4.loop_pause_sec", 0.5));

    enable_visualization_ = this->declare_parameter<bool>("enable_visualization", true);
    visualization_playback_enabled_ = enable_visualization_ && this->declare_parameter<bool>("enable_joint_state_playback", true);
    visualization_playback_speed_scale_ = std::max(1e-3, this->declare_parameter<double>("joint_state_playback_speed_scale", 1.0));

    robot_model_ = std::make_unique<RobotModel>(urdf_path, srdf_path, mesh_package_dirs);
    const auto& left_arm_group = robot_model_->getPlanningGroup(left_arm_group_name_);
    const auto& right_arm_group = robot_model_->getPlanningGroup(right_arm_group_name_);
    const auto& left_hand_group = robot_model_->getPlanningGroup(left_hand_group_name_);
    const auto& right_hand_group = robot_model_->getPlanningGroup(right_hand_group_name_);

    arm_home_positions_ = this->declare_parameter<std::vector<double>>(
      "dhz4.arm_home_positions", std::vector<double>(right_arm_group.joint_names.size(), 0.0));
    hand_open_positions_ = this->declare_parameter<std::vector<double>>(
      "dhz4.hand_open_positions", std::vector<double>{0.0, 0.0});
    hand_close_positions_ = this->declare_parameter<std::vector<double>>(
      "dhz4.hand_close_positions", std::vector<double>{-0.019, 0.019});

    if (arm_home_positions_.size() != right_arm_group.joint_names.size() ||
        arm_home_positions_.size() != left_arm_group.joint_names.size()) {
      throw std::runtime_error("dhz4.arm_home_positions size does not match arm group dof");
    }
    if (hand_open_positions_.size() != right_hand_group.joint_names.size() ||
        hand_open_positions_.size() != left_hand_group.joint_names.size()) {
      throw std::runtime_error("dhz4.hand_open_positions size does not match hand group dof");
    }
    if (hand_close_positions_.size() != right_hand_group.joint_names.size() ||
        hand_close_positions_.size() != left_hand_group.joint_names.size()) {
      throw std::runtime_error("dhz4.hand_close_positions size does not match hand group dof");
    }
    if (!mesh_scale_.empty() && mesh_scale_.size() != 3) {
      throw std::runtime_error("dhz4.mesh_scale must contain exactly 3 values");
    }

    arm_states_[0] = arm_home_positions_;
    arm_states_[1] = arm_home_positions_;
    hand_states_[0] = hand_open_positions_;
    hand_states_[1] = hand_open_positions_;
    selectArm(default_recatch_side_);

    plan_client_ = this->create_client<lite_motion_msgs::srv::PlanArmMotion>("plan_arm_motion");
    add_object_client_ = this->create_client<lite_motion_msgs::srv::AddSceneObject>("add_scene_object");
    attach_object_client_ = this->create_client<lite_motion_msgs::srv::AttachSceneObject>("attach_scene_object");
    remove_object_client_ = this->create_client<lite_motion_msgs::srv::RemoveSceneObject>("remove_scene_object");
    set_scene_collision_allowance_client_ = this->create_client<lite_motion_msgs::srv::SetSceneCollisionAllowance>("set_scene_collision_allowance");

    if (enable_visualization_) {
      rclcpp::QoS latched_qos(1);
      latched_qos.transient_local().reliable();
      ref_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "haomu_flow/reference_frames", latched_qos);
    }

    start_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      [this]() {
        start_timer_->cancel();
        publishReferenceMarkers();
        worker_ = std::thread([this]() { runSequence(); });
      });
  }

  ~HaomuFlowConsoleNode() override {
    stop_requested_.store(true);
    if (worker_.joinable()) {
      worker_.join();
    }
  }

private:
  std::string sideName(int side) const {
    return side == 0 ? "left" : "right";
  }

  void selectArm(int side) {
    if (side != 0 && side != 1) {
      throw std::runtime_error("arm side must be 0(left) or 1(right)");
    }
    saveActiveArmState();
    active_side_ = side;
    if (side == 0) {
      right_arm_group_ = left_arm_group_name_;
      right_hand_group_ = left_hand_group_name_;
      tcp_frame_ = left_tcp_frame_;
      ccb_frame_ = left_ccb_frame_;
      zc_relaxed_links_ = left_zc_relaxed_links_;
      return_home_relaxed_links_ = left_return_home_relaxed_links_;
      arc8_relaxed_links_ = left_arc8_relaxed_links_;
    } else {
      right_arm_group_ = right_arm_group_name_;
      right_hand_group_ = right_hand_group_name_;
      tcp_frame_ = right_tcp_frame_;
      ccb_frame_ = right_ccb_frame_;
      zc_relaxed_links_ = right_zc_relaxed_links_;
      return_home_relaxed_links_ = right_return_home_relaxed_links_;
      arc8_relaxed_links_ = right_arc8_relaxed_links_;
    }
    right_arm_state_ = arm_states_[static_cast<size_t>(side)];
    right_hand_state_ = hand_states_[static_cast<size_t>(side)];
  }

  void saveActiveArmState() {
    if (active_side_ != 0 && active_side_ != 1) {
      return;
    }
    arm_states_[static_cast<size_t>(active_side_)] = right_arm_state_;
    hand_states_[static_cast<size_t>(active_side_)] = right_hand_state_;
  }

  void initializeConsoleInput() {
    tty_input_.open("/dev/tty");
    if (tty_input_.is_open()) {
      input_ = &tty_input_;
    }
  }

  bool readInt(int& value) {
    return static_cast<bool>((*input_) >> value);
  }

  bool readString(std::string& value) {
    return static_cast<bool>((*input_) >> value);
  }

  std::array<double, 3> declareVec3Param(
    const std::string& name,
    const std::array<double, 3>& default_value) {
    auto values = this->declare_parameter<std::vector<double>>(
      name, std::vector<double>{default_value[0], default_value[1], default_value[2]});
    if (values.size() != 3) {
      throw std::runtime_error(name + " must contain exactly 3 values");
    }
    return {values[0], values[1], values[2]};
  }

  std::array<double, 2> declareVec2Param(
    const std::string& name,
    const std::array<double, 2>& default_value) {
    auto values = this->declare_parameter<std::vector<double>>(
      name, std::vector<double>{default_value[0], default_value[1]});
    if (values.size() != 2) {
      throw std::runtime_error(name + " must contain exactly 2 values");
    }
    return {wrapToPi(values[0]), wrapToPi(values[1])};
  }

  double yawSampleSpan() const {
    double span = wrapToPi(yaw_sample_range_[1] - yaw_sample_range_[0]);
    if (span < 0.0) {
      span += 2.0 * M_PI;
    }
    return std::max(0.0, span);
  }

  geometry_msgs::msg::PoseStamped makePoseStamped(
    const std::array<double, 3>& xyz,
    const std::array<double, 3>& rpy) const {
    geometry_msgs::msg::PoseStamped out;
    out.header.frame_id = planning_frame_;
    out.pose.position.x = xyz[0];
    out.pose.position.y = xyz[1];
    out.pose.position.z = xyz[2];
    out.pose.orientation = quaternionMsgFromRpy(rpy[0], rpy[1], rpy[2]);
    return out;
  }


  std::array<double, 3> zcTargetXyz(const std::string& name) const {
    if (name == "zc_f") return zc_f_xyz_;
    if (name == "zc_m") return zc_m_xyz_;
    if (name == "zc_l") return zc_l_xyz_;
    if (name == "zc_r") return zc_r_xyz_;
    RCLCPP_WARN(this->get_logger(), "Unknown dhz4.zc_target [%s], falling back to zc_f", name.c_str());
    return zc_f_xyz_;
  }

  std::array<double, 3> zcTargetRpy(const std::string& name) const {
    if (name == "zc_f") return zc_f_rpy_;
    if (name == "zc_m") return zc_m_rpy_;
    if (name == "zc_l") return zc_l_rpy_;
    if (name == "zc_r") return zc_r_rpy_;
    return zc_f_rpy_;
  }

  geometry_msgs::msg::PoseStamped zcTargetPose(const std::string& name) const {
    return makePoseStamped(zcTargetXyz(name), zcTargetRpy(name));
  }

  std::string zcObjectId(const std::string& name) const {
    // Keep the selected E3 under the canonical id [E3] so the legacy downstream
    // code that attaches E3 to tcp and uses E3 in relaxed collision lists remains unchanged.
    return name == zc_target_name_ ? e3_object_id_ : (e3_object_id_ + "_" + name);
  }

  std::vector<std::string> zcObjectIds() const {
    return {zcObjectId("zc_f"), zcObjectId("zc_m"), zcObjectId("zc_l"), zcObjectId("zc_r")};
  }

  std::vector<std::string> defaultRobotCollisionLinks() const {
    return {
      "base_link",
      "arm_left_J1_link", "arm_left_J2_link", "arm_left_J3_link", "arm_left_J4_link",
      "arm_left_J5_link", "arm_left_J6_link", "arm_left_J7_link",
      "camera_left_link", "zhua_left_1_link", "zhua_left_2_link",
      "arm_right_J1_link", "arm_right_J2_link", "arm_right_J3_link", "arm_right_J4_link",
      "arm_right_J5_link", "arm_right_J6_link", "arm_right_J7_link",
      "camera_right_link", "zhua_right_1_link", "zhua_right_2_link",
      "left_tcp", "left_ccb", "right_tcp", "right_ccb"};
  }

  geometry_msgs::msg::PoseStamped offsetAlongLocalAxis(
    const geometry_msgs::msg::PoseStamped& in,
    const Eigen::Vector3d& local_axis,
    double distance_m) const {
    geometry_msgs::msg::PoseStamped out = in;
    const Eigen::Quaterniond q = quaternionEigenFromMsg(in.pose.orientation);
    const Eigen::Vector3d delta = q.toRotationMatrix() * local_axis.normalized() * distance_m;
    out.pose.position.x += delta.x();
    out.pose.position.y += delta.y();
    out.pose.position.z += delta.z();
    return out;
  }

  geometry_msgs::msg::PoseStamped rotatePoseAboutLocalZ(
    const geometry_msgs::msg::PoseStamped& in,
    double yaw_offset_rad) const {
    geometry_msgs::msg::PoseStamped out = in;
    const Eigen::Quaterniond base_q = quaternionEigenFromMsg(in.pose.orientation);
    const Eigen::Quaterniond yaw_q(Eigen::AngleAxisd(yaw_offset_rad, Eigen::Vector3d::UnitZ()));
    const Eigen::Quaterniond q_new = (base_q * yaw_q).normalized();
    out.pose.orientation.x = q_new.x();
    out.pose.orientation.y = q_new.y();
    out.pose.orientation.z = q_new.z();
    out.pose.orientation.w = q_new.w();
    return out;
  }

  pinocchio::SE3 dhzWorldPose() const {
    return poseMsgToSE3(makePoseStamped(d_test_xyz_, d_test_rpy_).pose);
  }

  Eigen::Vector3d dhzLocalPointToWorld(const std::array<double, 3>& xyz_dhz) const {
    const auto dhz_world = dhzWorldPose();
    return dhz_world.translation() +
      dhz_world.rotation() * Eigen::Vector3d(xyz_dhz[0], xyz_dhz[1], xyz_dhz[2]);
  }

  Eigen::Vector3d dhzLocalAxisToWorld(const Eigen::Vector3d& axis_dhz) const {
    Eigen::Vector3d axis_world = dhzWorldPose().rotation() * axis_dhz;
    if (axis_world.norm() < 1e-12) {
      return Eigen::Vector3d::UnitZ();
    }
    return axis_world.normalized();
  }

  geometry_msgs::msg::PoseStamped rotatePoseAboutWorldAxis(
    const geometry_msgs::msg::PoseStamped& start_pose,
    const Eigen::Vector3d& center_world,
    const Eigen::Vector3d& axis_world,
    double angle_rad) const {
    geometry_msgs::msg::PoseStamped out = start_pose;
    const Eigen::AngleAxisd aa(angle_rad, axis_world.normalized());
    const Eigen::Matrix3d rot = aa.toRotationMatrix();

    const Eigen::Vector3d p0(
      start_pose.pose.position.x,
      start_pose.pose.position.y,
      start_pose.pose.position.z);
    const Eigen::Vector3d p = center_world + rot * (p0 - center_world);
    out.pose.position.x = p.x();
    out.pose.position.y = p.y();
    out.pose.position.z = p.z();

    const Eigen::Quaterniond q0 = quaternionEigenFromMsg(start_pose.pose.orientation);
    const Eigen::Quaterniond q_delta(aa);
    const Eigen::Quaterniond q_new = (q_delta * q0).normalized();
    out.pose.orientation.x = q_new.x();
    out.pose.orientation.y = q_new.y();
    out.pose.orientation.z = q_new.z();
    out.pose.orientation.w = q_new.w();
    return out;
  }

  size_t computeArcStepCount(
    double radius_m,
    double angle_rad,
    double cartesian_max_step_m,
    int min_step_count) const {
    const double arc_len = std::abs(radius_m * angle_rad);
    const size_t by_length = static_cast<size_t>(
      std::ceil(arc_len / std::max(1e-4, cartesian_max_step_m)));
    return std::max(static_cast<size_t>(std::max(1, min_step_count)), std::max<size_t>(1, by_length));
  }

  size_t computeArcStepCount(double radius_m, double angle_rad) const {
    return computeArcStepCount(radius_m, angle_rad, arc_cartesian_max_step_m_, arc_min_step_count_);
  }

  Eigen::VectorXd currentFullState() const {
    Eigen::VectorXd q_full = robot_model_->neutralConfiguration();

    const Eigen::VectorXd q_arm = Eigen::Map<const Eigen::VectorXd>(
      right_arm_state_.data(), static_cast<Eigen::Index>(right_arm_state_.size()));
    q_full = robot_model_->groupToFull(right_arm_group_, q_arm, q_full);

    const Eigen::VectorXd q_hand = Eigen::Map<const Eigen::VectorXd>(
      right_hand_state_.data(), static_cast<Eigen::Index>(right_hand_state_.size()));
    q_full = robot_model_->groupToFull(right_hand_group_, q_hand, q_full);

    return q_full;
  }

  pinocchio::SE3 currentFramePose(const std::string& frame_name) {
    const Eigen::VectorXd q_full = currentFullState();
    pinocchio::forwardKinematics(robot_model_->model(), robot_model_->data(), q_full);
    pinocchio::updateFramePlacements(robot_model_->model(), robot_model_->data());
    return robot_model_->data().oMf[robot_model_->getFrameIdChecked(frame_name)];
  }

  geometry_msgs::msg::PoseStamped currentFramePoseStamped(const std::string& frame_name) {
    geometry_msgs::msg::PoseStamped out;
    out.header.frame_id = planning_frame_;
    out.pose = poseMsgFromSE3(currentFramePose(frame_name));
    return out;
  }

  geometry_msgs::msg::Pose computeAttachRelativePose(
    const std::string& frame_name,
    const geometry_msgs::msg::PoseStamped& object_world_pose) {
    const pinocchio::SE3 frame_world = currentFramePose(frame_name);
    const pinocchio::SE3 object_world = poseMsgToSE3(object_world_pose.pose);
    return poseMsgFromSE3(frame_world.actInv(object_world));
  }

  geometry_msgs::msg::Pose makeE3RelativePoseInTcp(double yaw_rad = 0.0) const {
    geometry_msgs::msg::Pose relative_pose;
    relative_pose.orientation = quaternionMsgFromRpy(0.0, 0.0, yaw_rad);
    // E3 origin remains exactly coincident with tcp origin. The yaw tolerance is realized
    // only by spinning the E3 mesh/collision body around its own local Z axis.
    relative_pose.position.x = 0.0;
    relative_pose.position.y = 0.0;
    relative_pose.position.z = 0.0;
    return relative_pose;
  }

  geometry_msgs::msg::Pose makeFixedZzRelativePoseInTcp() const {
    geometry_msgs::msg::Pose relative_pose;
    relative_pose.orientation = quaternionMsgFromRpy(
      zz_attach_rpy_in_e3_[0],
      zz_attach_rpy_in_e3_[1],
      zz_attach_rpy_in_e3_[2]);

    const Eigen::Matrix3d r_tcp_zz = quaternionEigenFromMsg(relative_pose.orientation).toRotationMatrix();
    const Eigen::Vector3d p_tcp_contact(
      zz_attach_point_xyz_in_e3_[0],
      zz_attach_point_xyz_in_e3_[1],
      zz_attach_point_xyz_in_e3_[2]);
    const Eigen::Vector3d p_zz_contact(
      zz_attach_point_xyz_in_zz_[0],
      zz_attach_point_xyz_in_zz_[1],
      zz_attach_point_xyz_in_zz_[2]);

    // tcp_T_zz is chosen so that the fixed zz local attach point coincides with the
    // E3/tcp origin. It is intentionally independent from E3 yaw sampling: tcp and zz
    // must not rotate when E3 uses its allowed spin about the shared Z axis.
    const Eigen::Vector3d p_tcp_zz_origin = p_tcp_contact - r_tcp_zz * p_zz_contact;
    relative_pose.position.x = p_tcp_zz_origin.x();
    relative_pose.position.y = p_tcp_zz_origin.y();
    relative_pose.position.z = p_tcp_zz_origin.z();
    return relative_pose;
  }

  geometry_msgs::msg::PoseStamped composeFrameRelativePoseStamped(
    const std::string& frame_name,
    const geometry_msgs::msg::Pose& relative_pose) {
    geometry_msgs::msg::PoseStamped out;
    out.header.frame_id = planning_frame_;
    out.pose = poseMsgFromSE3(currentFramePose(frame_name) * poseMsgToSE3(relative_pose));
    return out;
  }

  bool attachZzToTcpWithE3ContactPoint(const std::string& stage_name) {
    const auto zz_relative_to_tcp = makeFixedZzRelativePoseInTcp();
    const auto zz_world_pose = composeFrameRelativePoseStamped(tcp_frame_, zz_relative_to_tcp);

    // The backend can attach scene objects only to robot frames. Because E3 origin is at
    // right_tcp origin, attaching zz to right_tcp with this fixed relative pose realizes
    // the required contact: zz local (0,0,-0.0275) sits on E3(0,0,0).
    if (!addMeshObject(zz_object_id_, zz_mesh_resource_, zz_world_pose)) {
      return false;
    }
    if (!setObjectAttached(zz_object_id_, true, tcp_frame_, &zz_relative_to_tcp)) {
      return false;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] fixed zz to tcp/E3 contact: zz local attach point=(%.4f, %.4f, %.4f) -> E3/tcp origin=(%.4f, %.4f, %.4f); zz_origin_in_tcp=(%.4f, %.4f, %.4f); E3 yaw remains independent",
      stage_name.c_str(),
      zz_attach_point_xyz_in_zz_[0], zz_attach_point_xyz_in_zz_[1], zz_attach_point_xyz_in_zz_[2],
      zz_attach_point_xyz_in_e3_[0], zz_attach_point_xyz_in_e3_[1], zz_attach_point_xyz_in_e3_[2],
      zz_relative_to_tcp.position.x, zz_relative_to_tcp.position.y, zz_relative_to_tcp.position.z);
    return true;
  }

  bool setE3RelativeYawInTcp(double yaw_rad, const std::string& stage_name) {
    const double normalized_yaw = wrapToPi(yaw_rad);
    const auto e3_relative_to_tcp = makeE3RelativePoseInTcp(normalized_yaw);
    if (!setObjectAttached(e3_object_id_, true, tcp_frame_, &e3_relative_to_tcp)) {
      return false;
    }
    current_e3_yaw_in_tcp_ = normalized_yaw;
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] set E3-only yaw about tcp/E3 local Z to %.4f rad; tcp, ccb, gripper and zz relative poses are unchanged",
      stage_name.c_str(),
      current_e3_yaw_in_tcp_);
    return true;
  }

  bool detachZzToWorldForYawSweep(const std::string& stage_name) {
    // Real scene detach: zz becomes a world object at its current world pose while
    // arm+tcp+E3 yaw samples are evaluated. This avoids rotating zz/tcp through the
    // sampled tolerance and keeps visualization tied to CollisionScene truth.
    if (!setObjectAttached(zz_object_id_, false, tcp_frame_)) {
      return false;
    }
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] temporarily detached zz as a world object for yaw sampling; E3 remains rigid on tcp",
      stage_name.c_str());
    return true;
  }

  bool reattachZzAfterYawSweep(const std::string& stage_name) {
    if (!attachZzToTcpWithE3ContactPoint(stage_name)) {
      return false;
    }
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] reattached zz after yaw sampling; collision and visualization now follow the true attached scene again",
      stage_name.c_str());
    return true;
  }

  geometry_msgs::msg::PoseStamped offsetCcbAlongDhzLocalAxis(
    const geometry_msgs::msg::PoseStamped& ccb_pose,
    const Eigen::Vector3d& dhz_local_axis,
    double distance_m) {
    geometry_msgs::msg::PoseStamped out = ccb_pose;
    const Eigen::Quaterniond q_dhz = quaternionEigenFromMsg(makePoseStamped(d_test_xyz_, d_test_rpy_).pose.orientation);
    const Eigen::Vector3d delta = q_dhz.toRotationMatrix() * dhz_local_axis.normalized() * distance_m;
    out.pose.position.x += delta.x();
    out.pose.position.y += delta.y();
    out.pose.position.z += delta.z();
    return out;
  }

  template<typename ServiceT>
  bool waitForService(
    const typename rclcpp::Client<ServiceT>::SharedPtr& client,
    const std::string& service_name) {
    if (client->wait_for_service(std::chrono::duration<double>(service_wait_sec_))) {
      return true;
    }
    RCLCPP_ERROR(this->get_logger(), "Service [%s] not available within %.2f s", service_name.c_str(), service_wait_sec_);
    return false;
  }

  template<typename ServiceT>
  std::shared_ptr<typename ServiceT::Response> callService(
    const typename rclcpp::Client<ServiceT>::SharedPtr& client,
    const std::shared_ptr<typename ServiceT::Request>& request,
    const std::string& service_name) {
    auto future = client->async_send_request(request);
    if (future.wait_for(std::chrono::duration<double>(response_wait_sec_)) != std::future_status::ready) {
      RCLCPP_ERROR(this->get_logger(), "Timed out waiting for service [%s] response after %.2f s", service_name.c_str(), response_wait_sec_);
      return nullptr;
    }
    return future.get();
  }

  bool addMeshObject(
    const std::string& object_id,
    const std::string& mesh_resource,
    const geometry_msgs::msg::PoseStamped& object_pose) {
    auto req = std::make_shared<lite_motion_msgs::srv::AddSceneObject::Request>();
    req->object_id = object_id;
    req->mesh_resource = mesh_resource;
    req->pose = object_pose;
    req->collision_pose = object_pose.pose;
    req->mesh_scale = mesh_scale_;

    auto resp = callService<lite_motion_msgs::srv::AddSceneObject>(add_object_client_, req, "add_scene_object");
    if (!resp || !resp->success) {
      RCLCPP_ERROR(this->get_logger(), "Add scene object [%s] failed: %s", object_id.c_str(), resp ? resp->message.c_str() : "no response");
      return false;
    }
    RCLCPP_INFO(this->get_logger(), "Scene object [%s] loaded from [%s]", object_id.c_str(), mesh_resource.c_str());
    sleepAfterAction(post_action_wait_sec_);
    return true;
  }

  bool removeObject(const std::string& object_id, bool warn_on_failure = true) {
    auto req = std::make_shared<lite_motion_msgs::srv::RemoveSceneObject::Request>();
    req->object_id = object_id;
    auto resp = callService<lite_motion_msgs::srv::RemoveSceneObject>(remove_object_client_, req, "remove_scene_object");
    if (!resp || !resp->success) {
      if (warn_on_failure) {
        RCLCPP_WARN(this->get_logger(), "Remove scene object [%s] failed or skipped: %s", object_id.c_str(), resp ? resp->message.c_str() : "no response");
      }
      return false;
    }
    sleepAfterAction(post_action_wait_sec_);
    return true;
  }

  bool setObjectAttached(
    const std::string& object_id,
    bool attach,
    const std::string& frame_id,
    const geometry_msgs::msg::Pose* relative_pose = nullptr) {
    auto req = std::make_shared<lite_motion_msgs::srv::AttachSceneObject::Request>();
    req->object_id = object_id;
    req->attach = attach;
    req->frame_id = frame_id;
    if (relative_pose != nullptr) {
      req->relative_pose = *relative_pose;
    } else {
      req->relative_pose.orientation.w = 1.0;
    }

    auto resp = callService<lite_motion_msgs::srv::AttachSceneObject>(attach_object_client_, req, "attach_scene_object");
    if (!resp || !resp->success) {
      RCLCPP_ERROR(this->get_logger(), "%s object [%s] failed: %s", attach ? "Attach" : "Detach", object_id.c_str(), resp ? resp->message.c_str() : "no response");
      return false;
    }
    RCLCPP_INFO(this->get_logger(), "%s [%s] %s [%s]", attach ? "Attached" : "Detached", object_id.c_str(), attach ? "to" : "from", frame_id.c_str());
    sleepAfterAction(post_action_wait_sec_);
    return true;
  }

  bool setRuntimeSceneCollisionAllowance(
    const std::vector<std::string>& object_ids,
    const std::vector<std::string>& link_names,
    double allowed_penetration_m,
    const std::string& stage_name) {
    auto req = std::make_shared<lite_motion_msgs::srv::SetSceneCollisionAllowance::Request>();
    req->object_ids = object_ids;
    req->link_names = link_names;
    req->allowed_penetration = allowed_penetration_m;

    auto resp = callService<lite_motion_msgs::srv::SetSceneCollisionAllowance>(
      set_scene_collision_allowance_client_, req, "set_scene_collision_allowance");
    if (!resp || !resp->success) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Set scene collision allowance failed at stage [%s]: %s",
        stage_name.c_str(),
        resp ? resp->message.c_str() : "no response");
      return false;
    }
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] applied runtime relaxed collision allowance: objects=%zu links=%zu allowed_penetration=%.4f m",
      stage_name.c_str(),
      object_ids.size(),
      link_names.size(),
      allowed_penetration_m);
    sleepAfterAction(post_action_wait_sec_);
    return true;
  }

  bool applyZzGlobalCollisionIgnore(const std::string& stage_name) {
    if (zz_global_ignore_links_.empty()) {
      return true;
    }
    return setRuntimeSceneCollisionAllowance(
      std::vector<std::string>{zz_object_id_},
      zz_global_ignore_links_,
      zz_global_ignore_allowed_penetration_m_,
      stage_name);
  }

  bool clearZzGlobalCollisionIgnore(const std::string& stage_name) {
    if (zz_global_ignore_links_.empty()) {
      return true;
    }
    return setRuntimeSceneCollisionAllowance(
      std::vector<std::string>{zz_object_id_},
      zz_global_ignore_links_,
      0.0,
      stage_name);
  }

  bool applyE3GlobalCollisionIgnore(const std::string& stage_name) {
    if (e3_global_ignore_links_.empty()) {
      return true;
    }
    return setRuntimeSceneCollisionAllowance(
      std::vector<std::string>{e3_object_id_},
      e3_global_ignore_links_,
      e3_global_ignore_allowed_penetration_m_,
      stage_name);
  }

  bool clearE3GlobalCollisionIgnore(const std::string& stage_name) {
    if (e3_global_ignore_links_.empty()) {
      return true;
    }
    return setRuntimeSceneCollisionAllowance(
      std::vector<std::string>{e3_object_id_},
      e3_global_ignore_links_,
      0.0,
      stage_name);
  }

  bool applyDhzGlobalCollisionIgnore(const std::string& stage_name) {
    if (dhz_global_ignore_links_.empty()) {
      return true;
    }
    return setRuntimeSceneCollisionAllowance(
      std::vector<std::string>{dhz_object_id_},
      dhz_global_ignore_links_,
      dhz_global_ignore_allowed_penetration_m_,
      stage_name);
  }

  bool clearDhzGlobalCollisionIgnore(const std::string& stage_name) {
    if (dhz_global_ignore_links_.empty()) {
      return true;
    }
    return setRuntimeSceneCollisionAllowance(
      std::vector<std::string>{dhz_object_id_},
      dhz_global_ignore_links_,
      0.0,
      stage_name);
  }

  double effectiveExecutionWaitSec(const trajectory_msgs::msg::JointTrajectory& traj) const {
    double wait_sec = trajectoryDurationSec(traj);
    if (visualization_playback_enabled_) {
      wait_sec /= visualization_playback_speed_scale_;
    }
    return wait_sec + post_action_wait_sec_ + 0.25;
  }

  bool executeArmPosePlan(
    const geometry_msgs::msg::PoseStamped& target_pose,
    const std::string& ee_frame,
    bool use_cartesian_path,
    const std::string& stage_name,
    double cartesian_max_step_override_m = -1.0,
    double cartesian_yaw_tolerance_rad = 0.0,
    int cartesian_yaw_sample_count = 1,
    geometry_msgs::msg::PoseStamped* executed_pose = nullptr,
    bool skip_cartesian_yaw_exact = false) {
    auto req = std::make_shared<lite_motion_msgs::srv::PlanArmMotion::Request>();
    req->group_name = right_arm_group_;
    req->ee_frame = ee_frame;
    req->use_pose_goal = true;
    req->target_pose = target_pose;
    req->start_joint_positions = right_arm_state_;
    req->use_cartesian_path = use_cartesian_path;
    req->cartesian_max_step =
      cartesian_max_step_override_m > 0.0 ? cartesian_max_step_override_m : cartesian_max_step_m_;
    if (use_cartesian_path && cartesian_use_sdls_) {
      req->use_request_solver_settings = true;
      req->request_use_sdls = true;
      req->request_cartesian_nullspace_gain = cartesian_nullspace_gain_;
      req->request_cartesian_singular_threshold = cartesian_singular_threshold_;
      req->request_cartesian_branch_jump_max_rad = cartesian_branch_jump_max_rad_;
      req->request_cartesian_branch_jump_norm_max = cartesian_branch_jump_norm_max_;
      req->request_cartesian_waypoint_pos_tolerance = cartesian_waypoint_pos_tolerance_;
      req->request_cartesian_waypoint_rot_tolerance = cartesian_waypoint_rot_tolerance_;
    }
    const bool use_yaw_tolerance =
      cartesian_yaw_tolerance_rad > 1e-6 && cartesian_yaw_sample_count > 1;
    if (use_yaw_tolerance) {
      const Eigen::Vector3d dhz_z_axis = dhzLocalAxisToWorld(Eigen::Vector3d::UnitZ());
      const Eigen::Quaterniond dhz_q(dhzWorldPose().rotation());
      req->use_cartesian_yaw_tolerance = true;
      req->cartesian_yaw_tolerance_rad = cartesian_yaw_tolerance_rad;
      req->cartesian_yaw_sample_count = cartesian_yaw_sample_count;
      req->cartesian_yaw_axis.x = dhz_z_axis.x();
      req->cartesian_yaw_axis.y = dhz_z_axis.y();
      req->cartesian_yaw_axis.z = dhz_z_axis.z();
      req->use_cartesian_yaw_target = false;
      req->cartesian_yaw_target_rad = 0.0;
      req->cartesian_yaw_reference_orientation.x = dhz_q.x();
      req->cartesian_yaw_reference_orientation.y = dhz_q.y();
      req->cartesian_yaw_reference_orientation.z = dhz_q.z();
      req->cartesian_yaw_reference_orientation.w = dhz_q.w();
      req->cartesian_yaw_target_frame = ee_frame;
      req->use_cartesian_yaw_reference_range = true;
      req->cartesian_yaw_reference_min_rad = yaw_sample_range_[0];
      req->cartesian_yaw_reference_max_rad = yaw_sample_range_[1];
      req->skip_cartesian_yaw_exact = skip_cartesian_yaw_exact;
    }

    auto resp = callService<lite_motion_msgs::srv::PlanArmMotion>(plan_client_, req, "plan_arm_motion");
    if (!resp || !resp->success) {
      RCLCPP_ERROR(this->get_logger(), "Arm planning failed at stage [%s]: %s", stage_name.c_str(), resp ? resp->message.c_str() : "no response");
      return false;
    }
    if (resp->trajectory.points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Arm planning returned an empty trajectory at stage [%s]", stage_name.c_str());
      return false;
    }

    if (executed_pose != nullptr) {
      *executed_pose = target_pose;
      if (resp->used_cartesian_yaw_tolerance && std::abs(resp->cartesian_yaw_offset_rad) > 1e-9) {
        const Eigen::Vector3d dhz_z_axis = dhzLocalAxisToWorld(Eigen::Vector3d::UnitZ());
        const Eigen::Vector3d center_world(
          executed_pose->pose.position.x,
          executed_pose->pose.position.y,
          executed_pose->pose.position.z);
        *executed_pose = rotatePoseAboutWorldAxis(
          *executed_pose,
          center_world,
          dhz_z_axis,
          resp->cartesian_yaw_offset_rad);
      }
    }

    right_arm_state_ = resp->trajectory.points.back().positions;
    const double wait_sec = effectiveExecutionWaitSec(resp->trajectory);
    RCLCPP_INFO(this->get_logger(), "Stage [%s] trajectory accepted, waiting %.3f s for execution/playback", stage_name.c_str(), wait_sec);
    sleepAfterAction(wait_sec);
    return true;
  }

  bool executeTcpPosePlanPositionFocused(
    const geometry_msgs::msg::PoseStamped& target_pose,
    const std::string& stage_name) {
    auto req = std::make_shared<lite_motion_msgs::srv::PlanArmMotion::Request>();
    req->group_name = right_arm_group_;
    req->ee_frame = tcp_frame_;
    req->use_pose_goal = true;
    req->target_pose = target_pose;
    req->start_joint_positions = right_arm_state_;
    req->use_cartesian_path = false;
    req->cartesian_max_step = 0.0;
    req->use_weighted_orientation_goal = true;
    req->orientation_weight_x = zc_orientation_weight_x_;
    req->orientation_weight_y = zc_orientation_weight_y_;
    req->orientation_weight_z = zc_orientation_weight_z_;
    req->use_request_solver_settings = true;
    req->request_ik_max_iterations = zc_ik_max_iterations_;
    req->request_ik_pos_tolerance = zc_ik_pos_tolerance_;
    req->request_ik_rot_tolerance = zc_ik_rot_tolerance_;
    req->request_ik_damping = zc_ik_damping_;
    req->request_ik_alpha = zc_ik_alpha_;
    req->request_ik_max_step_norm = zc_ik_max_step_norm_;
    req->request_ik_max_seed_count = zc_ik_max_seed_count_;
    req->request_planning_timeout = zc_planning_timeout_;
    req->request_path_check_step = zc_path_check_step_;
    req->request_ompl_max_plan_attempts = zc_ompl_max_plan_attempts_;
    req->request_max_ik_goal_candidates = zc_max_ik_goal_candidates_;

    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] using zc position-focused IK: TCP position strong, roll/pitch weights=[%.4f, %.4f], yaw weight=%.4f, pos_tol=%.3f, timeout=%.2f s; OMPL remains joint-space.",
      stage_name.c_str(),
      req->orientation_weight_x,
      req->orientation_weight_y,
      req->orientation_weight_z,
      zc_ik_pos_tolerance_,
      zc_planning_timeout_);

    auto resp = callService<lite_motion_msgs::srv::PlanArmMotion>(plan_client_, req, "plan_arm_motion");
    if (!resp || !resp->success) {
      RCLCPP_ERROR(this->get_logger(), "Position-focused TCP planning failed at stage [%s]: %s", stage_name.c_str(), resp ? resp->message.c_str() : "no response");
      return false;
    }
    if (resp->trajectory.points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Position-focused TCP planning returned an empty trajectory at stage [%s]", stage_name.c_str());
      return false;
    }

    right_arm_state_ = resp->trajectory.points.back().positions;
    const double wait_sec = effectiveExecutionWaitSec(resp->trajectory);
    RCLCPP_INFO(this->get_logger(), "Stage [%s] trajectory accepted, waiting %.3f s for execution/playback", stage_name.c_str(), wait_sec);
    sleepAfterAction(wait_sec);
    return true;
  }

  bool executeArmPosePlanWithLocalYawSweep(
    const geometry_msgs::msg::PoseStamped& target_pose,
    const std::string& ee_frame,
    bool use_cartesian_path,
    const std::string& stage_name,
    double cartesian_max_step_override_m,
    double yaw_tolerance_rad,
    int yaw_sample_count,
    geometry_msgs::msg::PoseStamped* executed_pose = nullptr,
    bool allow_zz_detach_for_yaw = false) {
    const bool use_yaw_tolerance = yaw_tolerance_rad > 1e-6 && yaw_sample_count > 1;
    const bool can_detach_zz_for_yaw =
      use_cartesian_path &&
      allow_zz_detach_for_yaw &&
      detach_zz_during_attached_chain_yaw_sweep_ &&
      use_yaw_tolerance;

    if (can_detach_zz_for_yaw) {
      if (executeArmPosePlan(
            target_pose,
            ee_frame,
            use_cartesian_path,
            stage_name,
            cartesian_max_step_override_m,
            0.0,
            1,
            executed_pose)) {
        return true;
      }
      if (!detachZzToWorldForYawSweep(stage_name + "_before_cartesian_yaw_tolerance")) {
        return false;
      }
      bool plan_ok = executeArmPosePlan(
        target_pose,
        ee_frame,
        use_cartesian_path,
        stage_name + "_detached_exact",
        cartesian_max_step_override_m,
        0.0,
        1,
        executed_pose,
        false);
      if (!plan_ok) {
        plan_ok = executeArmPosePlan(
          target_pose,
          ee_frame,
          use_cartesian_path,
          stage_name,
          cartesian_max_step_override_m,
          yaw_tolerance_rad,
          yaw_sample_count,
          executed_pose,
          false);
      }
      if (plan_ok) {
        if (!reattachZzAfterYawSweep(stage_name + "_reattach_after_success")) {
          return false;
        }
        return true;
      }
      (void)reattachZzAfterYawSweep(stage_name + "_reattach_after_failed_sweep");
      return false;
    }

    return executeArmPosePlan(
      target_pose,
      ee_frame,
      use_cartesian_path,
      stage_name,
      cartesian_max_step_override_m,
      yaw_tolerance_rad,
      yaw_sample_count,
      executed_pose);
  }

  bool executeCcbPosePlanWithYawSweep(
    const geometry_msgs::msg::PoseStamped& target_pose,
    bool use_cartesian_path,
    const std::string& stage_name,
    geometry_msgs::msg::PoseStamped* executed_pose = nullptr) {
    if (executeArmPosePlan(
          target_pose,
          ccb_frame_,
          use_cartesian_path,
          stage_name,
          -1.0,
          0.0,
          1,
          executed_pose)) {
      return true;
    }
    return executeArmPosePlan(
      target_pose,
      ccb_frame_,
      use_cartesian_path,
      stage_name,
      -1.0,
      yawSampleSpan(),
      yaw_sample_count_,
      executed_pose,
      !use_cartesian_path);
  }

  bool executeCcbArcAroundDhz(
    const std::array<double, 3>& center_xyz_dhz,
    const Eigen::Vector3d& axis_dhz,
    double angle_rad,
    const std::string& stage_name,
    double cartesian_max_step_m = -1.0,
    int min_step_count = -1,
    int yaw_sample_count_override = -1,
    double yaw_tolerance_rad = 0.0,
    bool allow_zz_detach_for_yaw = false) {
    auto rolling_pose = currentFramePoseStamped(ccb_frame_);
    const Eigen::Vector3d center_world = dhzLocalPointToWorld(center_xyz_dhz);
    const Eigen::Vector3d axis_world = dhzLocalAxisToWorld(axis_dhz);
    const Eigen::Vector3d start_pos(
      rolling_pose.pose.position.x,
      rolling_pose.pose.position.y,
      rolling_pose.pose.position.z);
    const double radius_m = (start_pos - center_world).norm();
    const double arc_step_m = cartesian_max_step_m > 0.0 ? cartesian_max_step_m : arc_cartesian_max_step_m_;
    const int arc_min_steps = min_step_count > 0 ? min_step_count : arc_min_step_count_;
    const int sweep_samples = yaw_sample_count_override > 0 ? yaw_sample_count_override : yaw_sample_count_;
    const double sweep_tolerance = std::max(0.0, yaw_tolerance_rad);
    const size_t step_count = computeArcStepCount(radius_m, angle_rad, arc_step_m, arc_min_steps);

    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] arc around dhz: center=(%.4f, %.4f, %.4f) radius=%.5f angle=%.4f rad steps=%zu max_step=%.4f yaw_tolerance=%.4f yaw_samples=%d",
      stage_name.c_str(),
      center_world.x(), center_world.y(), center_world.z(),
      radius_m,
      angle_rad,
      step_count,
      arc_step_m,
      sweep_tolerance,
      sweep_samples);

    for (size_t i = 1; i <= step_count; ++i) {
      const double theta_step = angle_rad / static_cast<double>(step_count);
      const auto waypoint =
        rotatePoseAboutWorldAxis(rolling_pose, center_world, axis_world, theta_step);
      const std::string waypoint_stage = stage_name + "_wp_" + std::to_string(i) + "_of_" + std::to_string(step_count);
      geometry_msgs::msg::PoseStamped executed_pose;
      if (!executeArmPosePlanWithLocalYawSweep(
            waypoint,
            ccb_frame_,
            true,
            waypoint_stage,
            arc_step_m,
            sweep_tolerance,
            sweep_samples,
            &executed_pose,
            allow_zz_detach_for_yaw)) {
        return false;
      }
      rolling_pose = executed_pose;
    }
    return true;
  }

  bool executeArmJointPlan(
    const std::vector<double>& target_joints,
    const std::string& stage_name,
    bool use_home_request_settings = false) {
    if (target_joints.size() == right_arm_state_.size()) {
      double max_err = 0.0;
      for (size_t i = 0; i < target_joints.size(); ++i) {
        max_err = std::max(max_err, std::abs(target_joints[i] - right_arm_state_[i]));
      }
      if (max_err < 1e-6) {
        RCLCPP_INFO(this->get_logger(), "Stage [%s] already at target arm joints, skipping planning.", stage_name.c_str());
        return true;
      }
    }

    auto req = std::make_shared<lite_motion_msgs::srv::PlanArmMotion::Request>();
    req->group_name = right_arm_group_;
    req->ee_frame = ccb_frame_;
    req->use_pose_goal = false;
    req->start_joint_positions = right_arm_state_;
    req->goal_joint_positions = target_joints;
    req->use_cartesian_path = false;
    req->cartesian_max_step = 0.0;
    if (use_home_request_settings) {
      req->use_request_solver_settings = true;
      req->request_planning_timeout = home_planning_timeout_;
      req->request_path_check_step = home_path_check_step_;
      req->request_ompl_max_plan_attempts = home_ompl_max_plan_attempts_;
      RCLCPP_INFO(
        this->get_logger(),
        "Stage [%s] using post-release home joint planning settings: timeout=%.2f s, path_step=%.3f, ompl_attempts=%d.",
        stage_name.c_str(),
        home_planning_timeout_,
        home_path_check_step_,
        home_ompl_max_plan_attempts_);
    }

    auto resp = callService<lite_motion_msgs::srv::PlanArmMotion>(plan_client_, req, "plan_arm_motion");
    if (!resp || !resp->success) {
      RCLCPP_ERROR(this->get_logger(), "Arm joint planning failed at stage [%s]: %s", stage_name.c_str(), resp ? resp->message.c_str() : "no response");
      return false;
    }
    if (resp->trajectory.points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Arm joint planning returned an empty trajectory at stage [%s]", stage_name.c_str());
      return false;
    }
    right_arm_state_ = resp->trajectory.points.back().positions;
    sleepAfterAction(effectiveExecutionWaitSec(resp->trajectory));
    return true;
  }

  bool executeHandJointPlan(const std::vector<double>& target_joints, const std::string& stage_name) {
    if (target_joints.size() == right_hand_state_.size()) {
      double max_err = 0.0;
      for (size_t i = 0; i < target_joints.size(); ++i) {
        max_err = std::max(max_err, std::abs(target_joints[i] - right_hand_state_[i]));
      }
      if (max_err < 1e-6) {
        RCLCPP_INFO(this->get_logger(), "Stage [%s] already at target hand joints, skipping planning.", stage_name.c_str());
        return true;
      }
    }

    auto req = std::make_shared<lite_motion_msgs::srv::PlanArmMotion::Request>();
    req->group_name = right_hand_group_;
    req->ee_frame = tcp_frame_;
    req->use_pose_goal = false;
    req->start_joint_positions = right_hand_state_;
    req->goal_joint_positions = target_joints;
    req->use_cartesian_path = false;
    req->cartesian_max_step = 0.0;
    auto resp = callService<lite_motion_msgs::srv::PlanArmMotion>(plan_client_, req, "plan_arm_motion");
    if (!resp || !resp->success) {
      RCLCPP_ERROR(this->get_logger(), "Hand planning failed at stage [%s]: %s", stage_name.c_str(), resp ? resp->message.c_str() : "no response");
      return false;
    }
    if (resp->trajectory.points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Hand planning returned an empty trajectory at stage [%s]", stage_name.c_str());
      return false;
    }
    right_hand_state_ = resp->trajectory.points.back().positions;
    sleepAfterAction(effectiveExecutionWaitSec(resp->trajectory));
    return true;
  }

  bool initializeSceneAtHome() {
    const auto d_test_pose = makePoseStamped(d_test_xyz_, d_test_rpy_);

    // New dhz_4 grasp setup: E3 is no longer born on the gripper/tcp.  Four E3
    // instances are placed at the four zc slots.  The selected zc object keeps the
    // canonical id [E3] so all downstream attach/relaxed-collision logic continues
    // to operate on the object that will be grasped.
    const std::vector<std::string> names{"zc_f", "zc_m", "zc_l", "zc_r"};
    for (const auto& name : names) {
      if (!addMeshObject(zcObjectId(name), e3_mesh_resource_, zcTargetPose(name))) {
        return false;
      }
    }
    current_e3_yaw_in_tcp_ = 0.0;

    // zz and dhz are both inserted at d_test. Scene-scene pairs are skipped by CollisionScene,
    // so they do not collide with each other but still collide with the robot/attached meshes.
    if (!addMeshObject(zz_object_id_, zz_mesh_resource_, d_test_pose)) {
      return false;
    }
    if (!addMeshObject(dhz_object_id_, dhz_mesh_resource_, d_test_pose)) {
      return false;
    }
    return true;
  }

  bool moveToSelectedZcAndAttachE3() {
    const auto target_pose = zcTargetPose(zc_target_name_);
    const auto all_zc_ids = zcObjectIds();

    if (zc_relaxed_allowed_penetration_m_ > 0.0) {
      if (!setRuntimeSceneCollisionAllowance(
            all_zc_ids,
            zc_relaxed_links_,
            zc_relaxed_allowed_penetration_m_,
            "before_move_tcp_to_" + zc_target_name_ + "_e3_zc_allowance")) {
        return false;
      }
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Moving tcp to selected zc slot [%s] to grasp E3. Four zc E3 objects are in the scene; selected object id=[%s].",
      zc_target_name_.c_str(),
      e3_object_id_.c_str());

    bool move_ok = false;
    if (zc_use_position_focused_goal_) {
      move_ok = executeTcpPosePlanPositionFocused(target_pose, "move_tcp_to_" + zc_target_name_ + "_for_e3_grasp");
    } else {
      move_ok = executeArmPosePlan(target_pose, tcp_frame_, false, "move_tcp_to_" + zc_target_name_ + "_for_e3_grasp");
    }
    if (!move_ok) {
      return false;
    }

    if (!executeHandJointPlan(hand_close_positions_, "close_gripper_at_" + zc_target_name_)) {
      return false;
    }

    // Preserve the original downstream semantics: E3 origin is fixed at the tcp origin.
    // The approach IK is position-focused, so yaw is free while the grasped tool frame
    // becomes the tcp-attached reference for all following zz/dhz motions.
    current_e3_yaw_in_tcp_ = 0.0;
    const auto e3_identity_on_tcp = makeE3RelativePoseInTcp(current_e3_yaw_in_tcp_);
    if (!setObjectAttached(e3_object_id_, true, tcp_frame_, &e3_identity_on_tcp)) {
      return false;
    }

    if (zc_relaxed_allowed_penetration_m_ > 0.0) {
      (void)setRuntimeSceneCollisionAllowance(
        all_zc_ids,
        zc_relaxed_links_,
        0.0,
        "after_attach_e3_clear_zc_allowance");
    }

    if (remove_unused_zc_e3_after_grasp_) {
      for (const auto& id : all_zc_ids) {
        if (id == e3_object_id_) {
          continue;
        }
        removeObject(id, false);
      }
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Selected zc E3 grasped and attached to [%s]; downstream zz/dhz motion keeps the original dhz_4 flow.",
      tcp_frame_.c_str());
    return true;
  }

  bool releaseObjectsAndReturnHome() {
    (void)clearZzGlobalCollisionIgnore("before_final_release_clear_zz_global_collision_ignore");
    (void)clearE3GlobalCollisionIgnore("before_final_release_clear_e3_global_collision_ignore");
    (void)clearDhzGlobalCollisionIgnore("before_final_release_clear_dhz_global_collision_ignore");

    // Release the gripper first, then break the visual/collision attachment relation.
    if (!executeHandJointPlan(hand_open_positions_, "open_gripper_after_dhz4_finish")) {
      return false;
    }

    // zz is attached during the legacy downstream flow.  Detach it before E3 so the
    // robot does not carry either scene object while returning to the all-zero posture.
    (void)setObjectAttached(zz_object_id_, false, tcp_frame_);
    if (!setObjectAttached(e3_object_id_, false, tcp_frame_)) {
      return false;
    }

    if (remove_zc_e3_objects_before_return_home_) {
      for (const auto& id : zcObjectIds()) {
        removeObject(id, false);
      }
      removeObject(zz_object_id_, false);
    }

    if (return_home_relaxed_allowed_penetration_m_ > 0.0) {
      if (!setRuntimeSceneCollisionAllowance(
            return_home_relaxed_object_ids_,
            return_home_relaxed_links_,
            return_home_relaxed_allowed_penetration_m_,
            "before_return_home_dhz_touch_allowance")) {
        return false;
      }
    }

    const bool home_ok = executeArmJointPlan(arm_home_positions_, "return_home_after_release", true);

    if (return_home_relaxed_allowed_penetration_m_ > 0.0) {
      (void)setRuntimeSceneCollisionAllowance(
        return_home_relaxed_object_ids_,
        return_home_relaxed_links_,
        0.0,
        "after_return_home_clear_dhz_touch_allowance");
    }

    return home_ok;
  }

  std::vector<std::string> occupiedZcObjectIds() const {
    std::vector<std::string> ids;
    for (size_t i = 0; i < zc_has_e3_.size(); ++i) {
      if (zc_has_e3_[i] && !zc_object_ids_[i].empty()) {
        ids.push_back(zc_object_ids_[i]);
      }
    }
    return ids;
  }

  int zcIndexFromInput(const std::string& input) const {
    if (input == "f") return 0;
    if (input == "m") return 1;
    if (input == "l") return 2;
    if (input == "r") return 3;
    return -1;
  }

  std::string zcNameFromIndex(int index) const {
    static const std::array<std::string, 4> names{"zc_f", "zc_m", "zc_l", "zc_r"};
    if (index < 0 || index >= static_cast<int>(names.size())) {
      return "zc_f";
    }
    return names[static_cast<size_t>(index)];
  }

  std::string makeCarryObjectId(int side) {
    return "E3_" + sideName(side) + "_carry_" + std::to_string(++e3_serial_);
  }

  int singleLoadedGripperSide() const {
    if (gripper_has_e3_[0] && !gripper_has_e3_[1]) return 0;
    if (!gripper_has_e3_[0] && gripper_has_e3_[1]) return 1;
    return -1;
  }

  bool waitAllServices() {
    return waitForService<lite_motion_msgs::srv::PlanArmMotion>(plan_client_, "plan_arm_motion") &&
      waitForService<lite_motion_msgs::srv::AddSceneObject>(add_object_client_, "add_scene_object") &&
      waitForService<lite_motion_msgs::srv::RemoveSceneObject>(remove_object_client_, "remove_scene_object") &&
      waitForService<lite_motion_msgs::srv::AttachSceneObject>(attach_object_client_, "attach_scene_object") &&
      waitForService<lite_motion_msgs::srv::SetSceneCollisionAllowance>(
        set_scene_collision_allowance_client_, "set_scene_collision_allowance");
  }

  bool initializeArmSide(int side) {
    selectArm(side);
    if (!executeArmJointPlan(arm_home_positions_, "initialize_" + sideName(side) + "_arm_home", true)) {
      return false;
    }
    if (!executeHandJointPlan(hand_open_positions_, "initialize_" + sideName(side) + "_open_gripper")) {
      return false;
    }
    saveActiveArmState();
    return true;
  }

  bool initializeStaticScene() {
    (void)clearZzGlobalCollisionIgnore("initialize_static_scene_clear_zz_global_collision_ignore");
    (void)clearE3GlobalCollisionIgnore("initialize_static_scene_clear_e3_global_collision_ignore");
    (void)clearDhzGlobalCollisionIgnore("initialize_static_scene_clear_dhz_global_collision_ignore");
    removeObject(zz_object_id_, false);
    removeObject(dhz_object_id_, false);
    const auto d_test_pose = makePoseStamped(d_test_xyz_, d_test_rpy_);
    if (!addMeshObject(zz_object_id_, zz_mesh_resource_, d_test_pose)) {
      return false;
    }
    if (!addMeshObject(dhz_object_id_, dhz_mesh_resource_, d_test_pose)) {
      return false;
    }
    return true;
  }

  bool runHaomuCatch(int side) {
    if (gripper_has_e3_[static_cast<size_t>(side)]) {
      throw std::runtime_error(sideName(side) + " gripper already has E3");
    }
    selectArm(side);
    e3_object_id_ = makeCarryObjectId(side);
    const auto object_pose = makePoseStamped(p_test_xyz_, p_test_rpy_);
    geometry_msgs::msg::PoseStamped grasp_tcp_pose = object_pose;

    if (!addMeshObject(e3_object_id_, e3_mesh_resource_, object_pose)) {
      return false;
    }
    if (!executeArmPosePlanWithLocalYawSweep(
          object_pose,
          tcp_frame_,
          false,
          "haomu_catch_move_tcp_to_p_test",
          -1.0,
          0.0,
          1,
          &grasp_tcp_pose)) {
      return false;
    }
    const auto attach_relative_pose = computeAttachRelativePose(tcp_frame_, object_pose);
    if (!setObjectAttached(e3_object_id_, true, tcp_frame_, &attach_relative_pose)) {
      return false;
    }
    if (!executeHandJointPlan(hand_close_positions_, "haomu_catch_close_gripper")) {
      return false;
    }
    const auto lift_pose = offsetAlongLocalAxis(grasp_tcp_pose, Eigen::Vector3d::UnitZ(), catch_lift_distance_m_);
    if (!executeArmPosePlan(lift_pose, tcp_frame_, true, "haomu_catch_lift_along_e3_z")) {
      return false;
    }

    gripper_has_e3_[static_cast<size_t>(side)] = true;
    gripper_object_ids_[static_cast<size_t>(side)] = e3_object_id_;
    saveActiveArmState();
    return true;
  }

  bool moveActiveTcpToZc(const std::string& zc_name, const std::string& stage_name) {
    const auto target_pose = zcTargetPose(zc_name);
    const auto occupied_ids = occupiedZcObjectIds();
    std::vector<std::string> relaxed_ids = occupied_ids;
    relaxed_ids.push_back(e3_object_id_);
    if (zc_relaxed_allowed_penetration_m_ > 0.0 && !relaxed_ids.empty()) {
      if (!setRuntimeSceneCollisionAllowance(
            relaxed_ids,
            zc_relaxed_links_,
            zc_relaxed_allowed_penetration_m_,
            "before_" + stage_name + "_zc_allowance")) {
        return false;
      }
    }

    bool move_ok = false;
    if (zc_use_position_focused_goal_) {
      move_ok = executeTcpPosePlanPositionFocused(target_pose, stage_name);
    } else {
      move_ok = executeArmPosePlan(target_pose, tcp_frame_, false, stage_name);
    }

    if (zc_relaxed_allowed_penetration_m_ > 0.0 && !relaxed_ids.empty()) {
      (void)setRuntimeSceneCollisionAllowance(
        relaxed_ids,
        zc_relaxed_links_,
        0.0,
        "after_" + stage_name + "_clear_zc_allowance");
    }
    return move_ok;
  }

  bool runHaomuStore(int side, int zc_index) {
    if (zc_has_e3_[static_cast<size_t>(zc_index)]) {
      throw std::runtime_error(zcNameFromIndex(zc_index) + " already has E3");
    }
    if (!gripper_has_e3_[static_cast<size_t>(side)]) {
      throw std::runtime_error(sideName(side) + " gripper has no E3");
    }
    selectArm(side);
    e3_object_id_ = gripper_object_ids_[static_cast<size_t>(side)];
    const std::string zc_name = zcNameFromIndex(zc_index);

    if (!moveActiveTcpToZc(zc_name, "haomu_store_move_tcp_to_" + zc_name)) {
      return false;
    }
    if (!executeHandJointPlan(hand_open_positions_, "haomu_store_open_gripper_at_" + zc_name)) {
      return false;
    }
    if (!setObjectAttached(e3_object_id_, false, tcp_frame_)) {
      return false;
    }

    zc_has_e3_[static_cast<size_t>(zc_index)] = true;
    zc_object_ids_[static_cast<size_t>(zc_index)] = e3_object_id_;
    gripper_has_e3_[static_cast<size_t>(side)] = false;
    gripper_object_ids_[static_cast<size_t>(side)].clear();
    saveActiveArmState();
    return true;
  }

  bool runHaomuRecatch(int side, int zc_index) {
    if (!zc_has_e3_[static_cast<size_t>(zc_index)]) {
      throw std::runtime_error(zcNameFromIndex(zc_index) + " has no E3");
    }
    if (gripper_has_e3_[static_cast<size_t>(side)]) {
      throw std::runtime_error(sideName(side) + " gripper already has E3");
    }
    selectArm(side);
    e3_object_id_ = zc_object_ids_[static_cast<size_t>(zc_index)];
    const std::string zc_name = zcNameFromIndex(zc_index);

    if (!moveActiveTcpToZc(zc_name, "haomu_recatch_move_tcp_to_" + zc_name)) {
      return false;
    }
    if (!executeHandJointPlan(hand_close_positions_, "haomu_recatch_close_gripper_at_" + zc_name)) {
      return false;
    }
    current_e3_yaw_in_tcp_ = 0.0;
    const auto e3_identity_on_tcp = makeE3RelativePoseInTcp(current_e3_yaw_in_tcp_);
    if (!setObjectAttached(e3_object_id_, true, tcp_frame_, &e3_identity_on_tcp)) {
      return false;
    }
    if (!executeArmJointPlan(arm_home_positions_, "haomu_recatch_return_p_init", true)) {
      return false;
    }

    zc_has_e3_[static_cast<size_t>(zc_index)] = false;
    zc_object_ids_[static_cast<size_t>(zc_index)].clear();
    gripper_has_e3_[static_cast<size_t>(side)] = true;
    gripper_object_ids_[static_cast<size_t>(side)] = e3_object_id_;
    saveActiveArmState();
    return true;
  }

  bool runDhzFromAttachedE3() {
    const auto d_test_pose = makePoseStamped(d_test_xyz_, d_test_rpy_);
    geometry_msgs::msg::PoseStamped reached_ccb_pose = d_test_pose;

    if (!initializeStaticScene()) {
      return false;
    }

    if (!executeArmPosePlan(d_test_pose, ccb_frame_, false, "move_ccb_to_d_test")) {
      return false;
    }
    reached_ccb_pose = currentFramePoseStamped(ccb_frame_);

    // At this point the arm has reached the zz area while only E3 is attached to TCP.
    // Semantic requirement: zz must NOT be fixed to E3/TCP before the first Cartesian
    // straight-line motion.  Keep zz as an independent world scene object during this
    // first straight approach, then connect zz to E3/TCP only after that motion has
    // completed and before the following push/rotation stages.
    const auto down_pose = offsetAlongLocalAxis(reached_ccb_pose, Eigen::Vector3d::UnitZ(), -ccb_approach_distance_m_);
    if (!executeArmPosePlanWithLocalYawSweep(
          down_pose,
          ccb_frame_,
          true,
          "move_ccb_along_negative_z_0p100_before_zz_attach",
          cartesian_max_step_m_,
          yawSampleSpan(),
          yaw_sample_count_,
          nullptr,
          false)) {
      return false;
    }

    // First Cartesian straight-line motion is complete.  Now create the visual and
    // collision attachment between zz and E3/TCP, just before the remaining Cartesian
    // push and the subsequent rotational actions.
    if (!attachZzToTcpWithE3ContactPoint("after_first_cartesian_before_rotation")) {
      return false;
    }

    if (!applyE3GlobalCollisionIgnore("before_move_ccb_along_attached_zz_positive_y_ignore_e3_all_collisions")) {
      return false;
    }
    if (!applyZzGlobalCollisionIgnore("before_move_ccb_along_attached_zz_positive_y_ignore_zz_all_collisions")) {
      return false;
    }
    if (!applyDhzGlobalCollisionIgnore("before_move_ccb_along_attached_zz_positive_y_ignore_dhz_all_collisions")) {
      return false;
    }

    const auto ccb_after_down_pose = currentFramePoseStamped(ccb_frame_);
    const auto push_pose = offsetCcbAlongDhzLocalAxis(ccb_after_down_pose, Eigen::Vector3d::UnitY(), zz_y_push_distance_m_);
    if (!executeArmPosePlanWithLocalYawSweep(
          push_pose,
          ccb_frame_,
          true,
          "move_ccb_along_attached_zz_positive_y_0p100",
          cartesian_max_step_m_,
          yawSampleSpan(),
          yaw_sample_count_,
          nullptr,
          true)) {
      return false;
    }

    if (!executeCcbArcAroundDhz(
          arc7_center_xyz_dhz_,
          Eigen::Vector3d::UnitX(),
          arc7_roll_angle_rad_,
          "arc7_ccb_rotate_negative_dhz_roll_1p57",
          -1.0,
          -1,
          yaw_sample_count_,
          yawSampleSpan(),
          true)) {
      return false;
    }

    if (!clearDhzGlobalCollisionIgnore("after_arc7_restore_dhz_collision_before_arc8")) {
      return false;
    }

    const double arc8_sign = (r_or_l_ == 0) ? -1.0 : 1.0;
    if (!setRuntimeSceneCollisionAllowance(
          arc8_relaxed_object_ids_,
          arc8_relaxed_links_,
          arc8_relaxed_allowed_penetration_m_,
          "before_arc8_ccb_rotate_dhz_yaw_by_r_or_l_1p57")) {
      return false;
    }
    if (!executeCcbArcAroundDhz(
          arc8_center_xyz_dhz_,
          Eigen::Vector3d::UnitZ(),
          arc8_sign * arc8_yaw_angle_abs_rad_,
          "arc8_ccb_rotate_dhz_yaw_by_r_or_l_1p57",
          arc8_cartesian_max_step_m_,
          arc8_min_step_count_,
          1,
          0.0,
          false)) {
      return false;
    }

    if (!releaseObjectsAndReturnHome()) {
      return false;
    }
    return true;
  }

  bool runHaomuDhz(int side) {
    if (!gripper_has_e3_[static_cast<size_t>(side)]) {
      throw std::runtime_error(sideName(side) + " gripper has no E3");
    }
    selectArm(side);
    e3_object_id_ = gripper_object_ids_[static_cast<size_t>(side)];
    arc8_relaxed_object_ids_ = {dhz_object_id_};

    if (!runDhzFromAttachedE3()) {
      return false;
    }
    gripper_has_e3_[static_cast<size_t>(side)] = false;
    gripper_object_ids_[static_cast<size_t>(side)].clear();
    saveActiveArmState();
    return true;
  }

  bool executeUserCommand(int command) {
    if (command == 1) {
      int side = 1;
      std::cout << "arm(0 left, 1 right): " << std::flush;
      if (!readInt(side)) {
        throw std::runtime_error("failed to read arm");
      }
      if (side != 0 && side != 1) {
        throw std::runtime_error("invalid arm");
      }
      return runHaomuCatch(side);
    }
    if (command == 2) {
      int side = 1;
      std::cout << "arm(0 left, 1 right): " << std::flush;
      if (!readInt(side)) {
        throw std::runtime_error("failed to read arm");
      }
      if (side != 0 && side != 1) {
        throw std::runtime_error("invalid arm");
      }
      std::string slot;
      std::cout << "zc(f/m/l/r): " << std::flush;
      if (!readString(slot)) {
        throw std::runtime_error("failed to read zc");
      }
      const int index = zcIndexFromInput(slot);
      if (index < 0) {
        throw std::runtime_error("invalid zc");
      }
      return runHaomuStore(side, index);
    }
    if (command == 3) {
      int side = 1;
      std::cout << "arm(0 left, 1 right): " << std::flush;
      if (!readInt(side)) {
        throw std::runtime_error("failed to read arm");
      }
      if (side != 0 && side != 1) {
        throw std::runtime_error("invalid arm");
      }
      std::string slot;
      std::cout << "zc(f/m/l/r): " << std::flush;
      if (!readString(slot)) {
        throw std::runtime_error("failed to read zc");
      }
      const int index = zcIndexFromInput(slot);
      if (index < 0) {
        throw std::runtime_error("invalid zc");
      }
      return runHaomuRecatch(side, index);
    }
    if (command == 4) {
      int side = 1;
      std::cout << "arm(0 left, 1 right): " << std::flush;
      if (!readInt(side)) {
        throw std::runtime_error("failed to read arm");
      }
      if (side != 0 && side != 1) {
        throw std::runtime_error("invalid arm");
      }
      return runHaomuDhz(side);
    }
    throw std::runtime_error("unknown command");
  }

  void runSequence() {
    if (!waitAllServices()) {
      std::cout << "异常: service unavailable" << std::endl;
      rclcpp::shutdown();
      return;
    }

    try {
      if (!initializeArmSide(0) || !initializeArmSide(1)) {
        std::cout << "异常: initialize failed" << std::endl;
        rclcpp::shutdown();
        return;
      }
    } catch (const std::exception& ex) {
      std::cout << "异常: " << ex.what() << std::endl;
      rclcpp::shutdown();
      return;
    }

    while (!stop_requested_.load() && rclcpp::ok()) {
      int command = 0;
      std::cout << "cmd(1 catch, 2 store, 3 recatch, 4 dhz, 0 quit): " << std::flush;
      if (!readInt(command) || command == 0) {
        break;
      }
      try {
        const bool ok = executeUserCommand(command);
        std::cout << (ok ? "OK" : "异常") << std::endl;
      } catch (const std::exception& ex) {
        std::cout << "异常: " << ex.what() << std::endl;
      }
    }

    saveActiveArmState();
    rclcpp::shutdown();
  }

  void bestEffortCleanup() {
    removeObject(zz_object_id_, false);
    removeObject(dhz_object_id_, false);
  }

  void sleepAfterAction(double seconds) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (!stop_requested_.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  void publishReferenceMarkers() {
    if (!ref_marker_pub_) {
      return;
    }
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(clear);
    int id = 0;
    appendFrameMarkers(arr, makePoseStamped(d_test_xyz_, d_test_rpy_), "d_test", id);
    appendFrameMarkers(arr, zcTargetPose("zc_f"), "zc_f", id);
    appendFrameMarkers(arr, zcTargetPose("zc_m"), "zc_m", id);
    appendFrameMarkers(arr, zcTargetPose("zc_l"), "zc_l", id);
    appendFrameMarkers(arr, zcTargetPose("zc_r"), "zc_r", id);
    appendPointMarker(arr, dhzLocalPointToWorld(arc7_center_xyz_dhz_), "arc7_center", id);
    appendPointMarker(arr, dhzLocalPointToWorld(arc8_center_xyz_dhz_), "arc8_center", id);
    ref_marker_pub_->publish(arr);
  }

  void appendPointMarker(
    visualization_msgs::msg::MarkerArray& arr,
    const Eigen::Vector3d& point_world,
    const std::string& ns,
    int& id) const {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = planning_frame_;
    marker.header.stamp = this->now();
    marker.ns = ns;
    marker.id = id++;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = point_world.x();
    marker.pose.position.y = point_world.y();
    marker.pose.position.z = point_world.z();
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.025;
    marker.scale.y = 0.025;
    marker.scale.z = 0.025;
    marker.color.r = 1.0f;
    marker.color.g = 0.7f;
    marker.color.b = 0.1f;
    marker.color.a = 1.0f;
    arr.markers.push_back(marker);

    visualization_msgs::msg::Marker label = marker;
    label.ns = ns + "_label";
    label.id = id++;
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.pose.position.z += 0.04;
    label.scale.x = 0.0;
    label.scale.y = 0.0;
    label.scale.z = 0.025;
    label.text = ns;
    arr.markers.push_back(label);
  }

  void appendFrameMarkers(
    visualization_msgs::msg::MarkerArray& arr,
    const geometry_msgs::msg::PoseStamped& pose,
    const std::string& ns,
    int& id) const {
    const Eigen::Quaterniond q = quaternionEigenFromMsg(pose.pose.orientation);
    const Eigen::Matrix3d rot = q.toRotationMatrix();
    const Eigen::Vector3d origin(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
    const std::array<Eigen::Vector3d, 3> axes = {
      rot * Eigen::Vector3d::UnitX(),
      rot * Eigen::Vector3d::UnitY(),
      rot * Eigen::Vector3d::UnitZ()};

    for (size_t i = 0; i < axes.size(); ++i) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = pose.header.frame_id;
      marker.header.stamp = this->now();
      marker.ns = ns + "_axis";
      marker.id = id++;
      marker.type = visualization_msgs::msg::Marker::ARROW;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.scale.x = 0.008;
      marker.scale.y = 0.015;
      marker.scale.z = 0.015;
      marker.color.a = 1.0;
      marker.color.r = (i == 0) ? 1.0f : 0.0f;
      marker.color.g = (i == 1) ? 1.0f : 0.0f;
      marker.color.b = (i == 2) ? 1.0f : 0.0f;
      geometry_msgs::msg::Point p0;
      p0.x = origin.x();
      p0.y = origin.y();
      p0.z = origin.z();
      geometry_msgs::msg::Point p1;
      p1.x = origin.x() + 0.06 * axes[i].x();
      p1.y = origin.y() + 0.06 * axes[i].y();
      p1.z = origin.z() + 0.06 * axes[i].z();
      marker.points.push_back(p0);
      marker.points.push_back(p1);
      arr.markers.push_back(marker);
    }

    visualization_msgs::msg::Marker label;
    label.header.frame_id = pose.header.frame_id;
    label.header.stamp = this->now();
    label.ns = ns + "_label";
    label.id = id++;
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    label.pose = pose.pose;
    label.pose.position.z += 0.05;
    label.scale.z = 0.03;
    label.color.r = 1.0f;
    label.color.g = 1.0f;
    label.color.b = 1.0f;
    label.color.a = 1.0f;
    label.text = ns;
    arr.markers.push_back(label);
  }

  std::unique_ptr<RobotModel> robot_model_;
  rclcpp::Client<lite_motion_msgs::srv::PlanArmMotion>::SharedPtr plan_client_;
  rclcpp::Client<lite_motion_msgs::srv::AddSceneObject>::SharedPtr add_object_client_;
  rclcpp::Client<lite_motion_msgs::srv::AttachSceneObject>::SharedPtr attach_object_client_;
  rclcpp::Client<lite_motion_msgs::srv::RemoveSceneObject>::SharedPtr remove_object_client_;
  rclcpp::Client<lite_motion_msgs::srv::SetSceneCollisionAllowance>::SharedPtr set_scene_collision_allowance_client_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ref_marker_pub_;
  rclcpp::TimerBase::SharedPtr start_timer_;

  std::thread worker_;
  std::atomic_bool stop_requested_{false};
  std::ifstream tty_input_;
  std::istream* input_{&std::cin};

  std::string planning_frame_;
  std::string left_arm_group_name_;
  std::string left_hand_group_name_;
  std::string left_tcp_frame_;
  std::string left_ccb_frame_;
  std::string right_arm_group_name_;
  std::string right_hand_group_name_;
  std::string right_tcp_frame_;
  std::string right_ccb_frame_;
  std::string right_arm_group_;
  std::string right_hand_group_;
  std::string tcp_frame_;
  std::string ccb_frame_;
  std::string e3_object_id_;
  std::string e3_mesh_resource_;
  std::string zz_object_id_;
  std::string zz_mesh_resource_;
  std::string dhz_object_id_;
  std::string dhz_mesh_resource_;
  std::vector<double> mesh_scale_;
  std::array<double, 3> d_test_xyz_{};
  std::array<double, 3> d_test_rpy_{};
  std::array<double, 3> p_test_xyz_{};
  std::array<double, 3> p_test_rpy_{};
  double catch_lift_distance_m_{0.11};
  int default_recatch_side_{1};
  std::array<double, 3> zc_f_xyz_{};
  std::array<double, 3> zc_f_rpy_{};
  std::array<double, 3> zc_m_xyz_{};
  std::array<double, 3> zc_m_rpy_{};
  std::array<double, 3> zc_l_xyz_{};
  std::array<double, 3> zc_l_rpy_{};
  std::array<double, 3> zc_r_xyz_{};
  std::array<double, 3> zc_r_rpy_{};
  std::string zc_target_name_{"zc_f"};
  bool remove_unused_zc_e3_after_grasp_{true};
  bool remove_zc_e3_objects_before_return_home_{true};
  bool zc_use_position_focused_goal_{true};
  double zc_orientation_weight_x_{0.015};
  double zc_orientation_weight_y_{0.015};
  double zc_orientation_weight_z_{0.0};
  int zc_ik_max_iterations_{260};
  double zc_ik_pos_tolerance_{0.003};
  double zc_ik_rot_tolerance_{0.15};
  double zc_ik_damping_{1.0e-4};
  double zc_ik_alpha_{0.35};
  double zc_ik_max_step_norm_{0.50};
  int zc_ik_max_seed_count_{14};
  double zc_planning_timeout_{1.2};
  double zc_path_check_step_{0.07};
  int zc_ompl_max_plan_attempts_{1};
  int zc_max_ik_goal_candidates_{4};
  double zc_relaxed_allowed_penetration_m_{0.005};
  std::vector<std::string> left_zc_relaxed_links_;
  std::vector<std::string> right_zc_relaxed_links_;
  std::vector<std::string> zc_relaxed_links_;
  double home_planning_timeout_{2.0};
  double home_path_check_step_{0.10};
  int home_ompl_max_plan_attempts_{3};
  double return_home_relaxed_allowed_penetration_m_{0.005};
  std::vector<std::string> return_home_relaxed_object_ids_;
  std::vector<std::string> left_return_home_relaxed_links_;
  std::vector<std::string> right_return_home_relaxed_links_;
  std::vector<std::string> return_home_relaxed_links_;
  double ccb_approach_distance_m_{0.100};
  double zz_y_push_distance_m_{0.100};
  std::array<double, 3> zz_attach_point_xyz_in_zz_{};
  std::array<double, 3> zz_attach_point_xyz_in_e3_{};
  std::array<double, 3> zz_attach_rpy_in_e3_{};
  std::array<double, 3> arc7_center_xyz_dhz_{};
  double arc7_roll_angle_rad_{-1.5707963267948966};
  std::array<double, 3> arc8_center_xyz_dhz_{};
  double arc8_yaw_angle_abs_rad_{1.5707963267948966};
  int r_or_l_{0};
  double cartesian_max_step_m_{0.08};
  bool cartesian_use_sdls_{true};
  double cartesian_nullspace_gain_{0.20};
  double cartesian_singular_threshold_{0.04};
  double cartesian_branch_jump_max_rad_{0.75};
  double cartesian_branch_jump_norm_max_{1.20};
  double cartesian_waypoint_pos_tolerance_{0.0015};
  double cartesian_waypoint_rot_tolerance_{0.020};
  double arc_cartesian_max_step_m_{0.12};
  int arc_min_step_count_{2};
  double arc8_cartesian_max_step_m_{0.12};
  int arc8_min_step_count_{3};
  std::array<double, 2> yaw_sample_range_{0.79, 2.3};
  int yaw_sample_count_{6};
  double arc8_relaxed_allowed_penetration_m_{0.010};
  std::vector<std::string> arc8_relaxed_object_ids_;
  std::vector<std::string> left_arc8_relaxed_links_;
  std::vector<std::string> right_arc8_relaxed_links_;
  std::vector<std::string> arc8_relaxed_links_;
  std::vector<std::string> zz_global_ignore_links_;
  double zz_global_ignore_allowed_penetration_m_{-1.0};
  std::vector<std::string> e3_global_ignore_links_;
  double e3_global_ignore_allowed_penetration_m_{-1.0};
  std::vector<std::string> dhz_global_ignore_links_;
  double dhz_global_ignore_allowed_penetration_m_{-1.0};
  bool detach_zz_during_attached_chain_yaw_sweep_{true};
  double current_e3_yaw_in_tcp_{0.0};
  double post_action_wait_sec_{0.05};
  double service_wait_sec_{30.0};
  double response_wait_sec_{30.0};
  bool enable_visualization_{true};
  bool visualization_playback_enabled_{true};
  double visualization_playback_speed_scale_{1.0};
  bool sequence_loop_{false};
  double loop_pause_sec_{0.5};
  std::vector<double> arm_home_positions_;
  std::vector<double> hand_open_positions_;
  std::vector<double> hand_close_positions_;
  std::vector<double> right_arm_state_;
  std::vector<double> right_hand_state_;
  std::array<std::vector<double>, 2> arm_states_;
  std::array<std::vector<double>, 2> hand_states_;
  int active_side_{-1};
  std::array<bool, 4> zc_has_e3_{false, false, false, false};
  std::array<std::string, 4> zc_object_ids_{};
  std::array<bool, 2> gripper_has_e3_{false, false};
  std::array<std::string, 2> gripper_object_ids_{};
  int e3_serial_{0};
};

}  // namespace lite_motion_planner

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<lite_motion_planner::HaomuFlowConsoleNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
