#include "lite_motion_planner/robot_model.hpp"

#include <lite_motion_msgs/srv/add_scene_object.hpp>
#include <lite_motion_msgs/srv/attach_scene_object.hpp>
#include <lite_motion_msgs/srv/plan_arm_motion.hpp>
#include <lite_motion_msgs/srv/remove_scene_object.hpp>
#include <lite_motion_msgs/srv/set_scene_collision_allowance.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <Eigen/Geometry>

#include <builtin_interfaces/msg/duration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
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

double pointTimeSec(const trajectory_msgs::msg::JointTrajectoryPoint& point) {
  return static_cast<double>(point.time_from_start.sec) +
         static_cast<double>(point.time_from_start.nanosec) * 1e-9;
}

builtin_interfaces::msg::Duration durationFromSec(double seconds) {
  builtin_interfaces::msg::Duration out;
  if (seconds < 0.0) {
    seconds = 0.0;
  }
  out.sec = static_cast<int32_t>(std::floor(seconds));
  out.nanosec = static_cast<uint32_t>(
    std::llround((seconds - static_cast<double>(out.sec)) * 1e9));
  if (out.nanosec >= 1000000000u) {
    out.sec += 1;
    out.nanosec -= 1000000000u;
  }
  return out;
}

}  // namespace

class Dhz4Node : public rclcpp::Node {
public:
  Dhz4Node()
  : Node("dhz_4") {
    const auto urdf_path = this->declare_parameter<std::string>("urdf_path", "");
    const auto srdf_path = this->declare_parameter<std::string>("srdf_path", "");
    const auto mesh_package_dirs = this->declare_parameter<std::vector<std::string>>(
      "mesh_package_dirs", std::vector<std::string>());

    planning_frame_ = this->declare_parameter<std::string>("planning_frame", "base_link");
    arm_side_ = normalizeArmSide(this->declare_parameter<std::string>("dhz4.arm_side", "right"));
    left_arm_group_name_ = this->declare_parameter<std::string>("dhz4.left_arm_group", "left_arm");
    left_hand_group_name_ = this->declare_parameter<std::string>("dhz4.left_hand_group", "left_hand");
    left_tcp_frame_ = this->declare_parameter<std::string>("dhz4.left_tcp_frame", "left_tcp");
    left_ccb_frame_ = this->declare_parameter<std::string>("dhz4.left_ccb_frame", "left_ccb");
    right_arm_group_name_ = this->declare_parameter<std::string>("dhz4.right_arm_group", "right_arm");
    right_hand_group_name_ = this->declare_parameter<std::string>("dhz4.right_hand_group", "right_hand");
    const auto legacy_right_tcp_frame = this->declare_parameter<std::string>("dhz4.tcp_frame", "right_tcp");
    const auto legacy_right_ccb_frame = this->declare_parameter<std::string>("dhz4.ccb_frame", "right_ccb");
    right_tcp_frame_ = this->declare_parameter<std::string>("dhz4.right_tcp_frame", legacy_right_tcp_frame);
    right_ccb_frame_ = this->declare_parameter<std::string>("dhz4.right_ccb_frame", legacy_right_ccb_frame);
    left_camera_frame_ = this->declare_parameter<std::string>("dhz4.left_camera_frame", "camera_left_link");
    right_camera_frame_ = this->declare_parameter<std::string>("dhz4.right_camera_frame", "camera_right_link");
    selectArm(arm_side_);
    d_test_frame_ = this->declare_parameter<std::string>(
      "dhz4.d_test_frame",
      arm_side_ == "left" ? left_camera_frame_ : right_camera_frame_);

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
    zc_target_name_ = this->declare_parameter<std::string>(
      "dhz4.zc_target",
      arm_side_ == "left" ? "zc_m" : "zc_r");
    validateZcTargetForArm();

    fixed_tail_trajectory_dir_ = this->declare_parameter<std::string>(
      "dhz4.fixed_tail_trajectory_dir",
      "/home/zwt/lite_motion_ws/src/lite_motion_planner/config");
    fixed_tail_trajectory_file_ = this->declare_parameter<std::string>(
      "dhz4.fixed_tail_trajectory_file", "");
    if (fixed_tail_trajectory_file_.empty()) {
      fixed_tail_trajectory_file_ = makeDefaultFixedHeadTrajectoryFile();
    }
    if (!fixed_tail_trajectory_file_.empty()) {
      const std::filesystem::path configured_path(fixed_tail_trajectory_file_);
      if (!configured_path.is_absolute()) {
        fixed_tail_trajectory_file_ =
          (std::filesystem::path(fixed_tail_trajectory_dir_) / configured_path).string();
      }
    }

    home_planning_timeout_ = std::max(0.05, this->declare_parameter<double>("dhz4.home_planning_timeout", 2.0));
    home_path_check_step_ = std::max(0.005, this->declare_parameter<double>("dhz4.home_path_check_step", 0.10));
    home_ompl_max_plan_attempts_ = std::max<int>(1, this->declare_parameter<int>("dhz4.home_ompl_max_plan_attempts", 3));
    base_group_collision_links_ = this->declare_parameter<std::vector<std::string>>(
      "dhz4.base_group_collision_links",
      std::vector<std::string>{"base_link", "cun_1_link", "cun_2_link", "cun_3_link", "cun_4_link"});
    base_group_collision_allowed_penetration_m_ = std::max(
      0.0,
      this->declare_parameter<double>(
        "dhz4.base_group_collision_allowed_penetration", 0.010));
    gripper_e3_ignore_allowed_penetration_m_ = this->declare_parameter<double>(
      "dhz4.gripper_e3_ignore_allowed_penetration", -1.0);
    return_home_self_collision_allowed_penetration_m_ = std::max(
      0.0,
      this->declare_parameter<double>(
        "dhz4.return_home_self_collision_allowed_penetration", 0.001));
    return_home_e3_allowed_penetration_m_ = std::max(
      0.0,
      this->declare_parameter<double>(
        "dhz4.return_home_e3_collision_allowed_penetration", 0.002));
    frozen_d_test_scene_allowed_penetration_m_ = std::max(
      0.0,
      this->declare_parameter<double>(
        "dhz4.frozen_d_test_scene_allowed_penetration", 0.002));
    arc7_dhz_allowed_penetration_m_ = std::max(
      0.0,
      this->declare_parameter<double>(
        "dhz4.arc7_dhz_collision_allowed_penetration", 0.006));

    const double legacy_d_test_pre_offset =
      this->declare_parameter<double>(
        "dhz4.d_test_pre_offset_along_zz_negative_z_m",
        std::numeric_limits<double>::quiet_NaN());
    if (std::isfinite(legacy_d_test_pre_offset)) {
      RCLCPP_WARN(
        this->get_logger(),
        "dhz4.d_test_pre_offset_along_zz_negative_z_m is deprecated; use dhz4.d_test_pre_offset_along_zz_positive_z_m. The value %.4f is interpreted as a +Z pre-offset magnitude.",
        legacy_d_test_pre_offset);
    }
    const double configured_d_test_pre_offset =
      this->declare_parameter<double>(
        "dhz4.d_test_pre_offset_along_zz_positive_z_m",
        std::isfinite(legacy_d_test_pre_offset) ? std::abs(legacy_d_test_pre_offset) : 0.012);
    d_test_pre_offset_along_zz_positive_z_m_ = std::abs(configured_d_test_pre_offset);
    if (configured_d_test_pre_offset < 0.0) {
      RCLCPP_WARN(
        this->get_logger(),
        "dhz4.d_test_pre_offset_along_zz_positive_z_m is negative (%.4f); using its magnitude %.4f along dhz local +Z.",
        configured_d_test_pre_offset,
        d_test_pre_offset_along_zz_positive_z_m_);
    }
    ccb_approach_distance_m_ = this->declare_parameter<double>("dhz4.ccb_approach_distance_m", 0.113);
    zz_y_push_distance_m_ = this->declare_parameter<double>("dhz4.zz_y_push_distance_m", 0.100);
    zz_attach_point_xyz_in_zz_ = declareVec3Param("dhz4.zz_attach_point_xyz_in_zz", {0.0, 0.0, -0.0275});
    zz_attach_point_xyz_in_e3_ = declareVec3Param("dhz4.zz_attach_point_xyz_in_e3", {0.0, 0.0, 0.0});
    zz_attach_rpy_in_e3_ = declareVec3Param("dhz4.zz_attach_rpy_in_e3", {0.0, 0.0, 0.0});
    arc7_center_xyz_dhz_ = declareVec3Param("dhz4.arc7_center_xyz_dhz", {0.0, 0.208, -0.169});
    arc7_roll_angle_rad_ = this->declare_parameter<double>("dhz4.arc7_roll_angle_rad", -1.5707963267948966);
    arc7_pre_solve_d2_enabled_ = this->declare_parameter<bool>("dhz4.arc7_pre_solve_d2_enabled", true);
    arc7_d2_start_xyz_dhz_ = declareVec3Param("dhz4.arc7_d2_start_xyz_dhz", {0.0, 0.1, -0.1});
    arc7_d2_start_rpy_dhz_ = declareVec3Param("dhz4.arc7_d2_start_rpy_dhz", {0.0, 0.0, 0.0});
    arc7_d2_yaw_sample_range_ = declareVec2Param("dhz4.arc7_d2_yaw_sample_range", {-0.79, 2.3});
    arc7_d2_yaw_sample_count_ = std::max<int>(
      1,
      this->declare_parameter<int>("dhz4.arc7_d2_yaw_sample_count", 8));
    arc7_d2_cartesian_max_step_m_ = std::max(
      1e-4,
      this->declare_parameter<double>("dhz4.arc7_d2_cartesian_max_step_m", 0.08));
    arc7_d2_min_step_count_ = std::max<int>(
      1,
      this->declare_parameter<int>("dhz4.arc7_d2_min_step_count", 2));
    arc7_d2_refinement_max_steps_ = std::max<int>(
      arc7_d2_min_step_count_,
      this->declare_parameter<int>("dhz4.arc7_d2_refinement_max_steps", 4));
    arc7_d2_ik_max_iterations_ = std::max<int>(
      20,
      this->declare_parameter<int>("dhz4.arc7_d2_ik_max_iterations", 180));
    arc7_d2_ik_pos_tolerance_ = std::max(
      1e-5,
      this->declare_parameter<double>("dhz4.arc7_d2_ik_pos_tolerance", 0.010));
    arc7_d2_ik_rot_tolerance_ = std::max(
      1e-5,
      this->declare_parameter<double>("dhz4.arc7_d2_ik_rot_tolerance", 0.180));
    arc7_d2_ik_damping_ = std::max(
      1e-12,
      this->declare_parameter<double>("dhz4.arc7_d2_ik_damping", 1.5e-3));
    arc7_d2_ik_alpha_ = std::max(
      1e-3,
      this->declare_parameter<double>("dhz4.arc7_d2_ik_alpha", 0.90));
    arc7_d2_ik_max_step_norm_ = std::max(
      1e-3,
      this->declare_parameter<double>("dhz4.arc7_d2_ik_max_step_norm", 2.50));
    arc7_d2_ik_max_seed_count_ = std::max<int>(
      1,
      this->declare_parameter<int>("dhz4.arc7_d2_ik_max_seed_count", 48));
    arc7_d2_max_ik_goal_candidates_ = std::max<int>(
      1,
      this->declare_parameter<int>("dhz4.arc7_d2_max_ik_goal_candidates", 12));
    arc7_d2_planning_timeout_ = std::max(
      0.05,
      this->declare_parameter<double>("dhz4.arc7_d2_planning_timeout", 0.80));
    arc7_d2_path_check_step_ = std::max(
      0.001,
      this->declare_parameter<double>("dhz4.arc7_d2_path_check_step", 0.030));
    arc7_d2_waypoint_pos_tolerance_ = std::max(
      1e-6,
      this->declare_parameter<double>("dhz4.arc7_d2_waypoint_pos_tolerance", 0.006));
    arc7_d2_waypoint_rot_tolerance_ = std::max(
      1e-6,
      this->declare_parameter<double>("dhz4.arc7_d2_waypoint_rot_tolerance", 0.120));
    arc8_center_xyz_dhz_ = declareVec3Param("dhz4.arc8_center_xyz_dhz", {0.0, 0.223, -0.064});
    arc8_yaw_angle_abs_rad_ = std::abs(this->declare_parameter<double>("dhz4.arc8_yaw_angle_abs_rad", 1.5707963267948966));
    r_or_l_ = this->declare_parameter<int>("dhz4.r_or_l", 0);
    const double legacy_straight_cartesian_max_step =
      this->declare_parameter<double>("dhz4.cartesian_max_step_m", 0.025);
    straight_cartesian_max_step_m_ = std::max(
      1e-4,
      this->declare_parameter<double>(
        "dhz4.straight_cartesian_max_step_m",
        legacy_straight_cartesian_max_step));
    const double legacy_arc7_cartesian_max_step =
      this->declare_parameter<double>("dhz4.arc_cartesian_max_step_m", 0.042);
    arc7_cartesian_max_step_m_ = std::max(
      1e-4,
      this->declare_parameter<double>(
        "dhz4.arc7_cartesian_max_step_m",
        legacy_arc7_cartesian_max_step));
    const int legacy_min_step_count =
      this->declare_parameter<int>("dhz4.arc_min_step_count", 4);
    min_step_count_ = std::max<int>(
      1,
      this->declare_parameter<int>("dhz4.min_step_count", legacy_min_step_count));
    refinement_max_steps_ = std::max<int>(
      min_step_count_,
      this->declare_parameter<int>("dhz4.refinement_max_steps", 8));
    arc7_refinement_max_steps_ = std::max<int>(
      min_step_count_,
      this->declare_parameter<int>("dhz4.arc7_refinement_max_steps", refinement_max_steps_));
    // dhz_4 uses dhz-local Z yaw tolerance on the pre-offset pose-goal and on the
    // Cartesian straight/arc stages that follow; each stage keeps its own sample budget.
    arc8_cartesian_max_step_m_ = std::max(1e-4, this->declare_parameter<double>("dhz4.arc8_cartesian_max_step_m", 0.004));
    arc8_min_step_count_ = std::max<int>(1, this->declare_parameter<int>("dhz4.arc8_min_step_count", 10));
    arc8_refinement_max_steps_ = std::max<int>(
      arc8_min_step_count_,
      this->declare_parameter<int>("dhz4.arc8_refinement_max_steps", 14));
    yaw_sample_range_ = declareVec2Param("dhz4.yaw_sample_range", {0.79, 2.3});
    yaw_sample_count_ = std::max<int>(1, this->declare_parameter<int>("dhz4.yaw_sample_count", 6));
    cartesian_use_request_solver_settings_ = this->declare_parameter<bool>(
      "dhz4.cartesian_use_request_solver_settings", true);
    cartesian_use_sdls_ = this->declare_parameter<bool>(
      "dhz4.cartesian_use_sdls", true);
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
    cartesian_ik_max_iterations_ = std::max<int>(
      50,
      this->declare_parameter<int>("dhz4.cartesian_ik_max_iterations", 1800));
    cartesian_ik_pos_tolerance_ = std::max(
      1e-5,
      this->declare_parameter<double>("dhz4.cartesian_ik_pos_tolerance", 0.002));
    cartesian_ik_rot_tolerance_ = std::max(
      1e-5,
      this->declare_parameter<double>("dhz4.cartesian_ik_rot_tolerance", 0.08));
    cartesian_ik_damping_ = std::max(
      1e-12,
      this->declare_parameter<double>("dhz4.cartesian_ik_damping", 1.5e-3));
    cartesian_ik_alpha_ = std::max(
      1e-3,
      this->declare_parameter<double>("dhz4.cartesian_ik_alpha", 0.55));
    cartesian_ik_max_step_norm_ = std::max(
      1e-3,
      this->declare_parameter<double>("dhz4.cartesian_ik_max_step_norm", 0.20));
    cartesian_ik_max_seed_count_ = std::max<int>(
      1,
      this->declare_parameter<int>("dhz4.cartesian_ik_max_seed_count", 60));
    cartesian_planning_timeout_ = std::max(
      0.05,
      this->declare_parameter<double>("dhz4.cartesian_planning_timeout", 1.5));
    cartesian_path_check_step_ = std::max(
      0.001,
      this->declare_parameter<double>("dhz4.cartesian_path_check_step", 0.025));
    detach_zz_during_attached_chain_yaw_sweep_ = this->declare_parameter<bool>("dhz4.detach_zz_during_attached_chain_yaw_sweep", true);
    detach_zz_during_attached_chain_motion_ = this->declare_parameter<bool>("dhz4.detach_zz_during_attached_chain_motion", false);
    if (detach_zz_during_attached_chain_motion_) {
      RCLCPP_WARN(
        this->get_logger(),
        "dhz4.detach_zz_during_attached_chain_motion is deprecated and ignored; zz is detached only for nonzero yaw candidates.");
    }
    post_action_wait_sec_ = this->declare_parameter<double>("dhz4.post_action_wait_sec", 0.05);
    service_wait_sec_ = this->declare_parameter<double>("dhz4.service_wait_sec", 30.0);
    response_wait_sec_ = this->declare_parameter<double>("dhz4.response_wait_sec", 30.0);
    sequence_loop_ = this->declare_parameter<bool>("dhz4.sequence_loop", false);
    loop_pause_sec_ = std::max(0.0, this->declare_parameter<double>("dhz4.loop_pause_sec", 0.5));

    enable_visualization_ = this->declare_parameter<bool>("enable_visualization", true);
    visualization_playback_enabled_ = enable_visualization_ && this->declare_parameter<bool>("enable_joint_state_playback", true);
    visualization_playback_speed_scale_ = std::max(1e-3, this->declare_parameter<double>("joint_state_playback_speed_scale", 1.0));

    robot_model_ = std::make_unique<RobotModel>(urdf_path, srdf_path, mesh_package_dirs);
    const auto& arm_group = robot_model_->getPlanningGroup(right_arm_group_);
    const auto& hand_group = robot_model_->getPlanningGroup(right_hand_group_);

    arm_home_positions_ = this->declare_parameter<std::vector<double>>(
      "dhz4.arm_home_positions", std::vector<double>(arm_group.joint_names.size(), 0.0));
    hand_open_positions_ = this->declare_parameter<std::vector<double>>(
      "dhz4.hand_open_positions", std::vector<double>{0.0, 0.0});
    hand_close_positions_ = this->declare_parameter<std::vector<double>>(
      "dhz4.hand_close_positions", std::vector<double>{-0.019, 0.019});
    right_observe_joint_positions_ = this->declare_parameter<std::vector<double>>(
      "dhz4.right_observe_joint_positions",
      std::vector<double>{0.40, 0.00, 1.57, 0.40, 2.62, 0.00, -0.50, -0.02, 0.02});
    left_observe_joint_positions_ = this->declare_parameter<std::vector<double>>(
      "dhz4.left_observe_joint_positions",
      std::vector<double>{0.40, 0.00, 1.57, -0.40, 2.62, 0.00, -0.50, -0.02, 0.02});

    if (arm_home_positions_.size() != arm_group.joint_names.size()) {
      throw std::runtime_error("dhz4.arm_home_positions size does not match selected arm group dof");
    }
    if (hand_open_positions_.size() != hand_group.joint_names.size()) {
      throw std::runtime_error("dhz4.hand_open_positions size does not match selected hand group dof");
    }
    if (hand_close_positions_.size() != hand_group.joint_names.size()) {
      throw std::runtime_error("dhz4.hand_close_positions size does not match selected hand group dof");
    }
    const size_t observe_joint_dim = arm_group.joint_names.size() + hand_group.joint_names.size();
    if (right_observe_joint_positions_.size() != observe_joint_dim) {
      throw std::runtime_error("dhz4.right_observe_joint_positions size must match selected arm+hand dof");
    }
    if (left_observe_joint_positions_.size() != observe_joint_dim) {
      throw std::runtime_error("dhz4.left_observe_joint_positions size must match selected arm+hand dof");
    }
    const auto& selected_observe_positions =
      arm_side_ == "left" ? left_observe_joint_positions_ : right_observe_joint_positions_;
    observe_arm_positions_.assign(
      selected_observe_positions.begin(),
      selected_observe_positions.begin() + arm_group.joint_names.size());
    observe_hand_positions_.assign(
      selected_observe_positions.begin() + arm_group.joint_names.size(),
      selected_observe_positions.end());
    if (!mesh_scale_.empty() && mesh_scale_.size() != 3) {
      throw std::runtime_error("dhz4.mesh_scale must contain exactly 3 values");
    }

    right_arm_state_ = arm_home_positions_;
    // Start as open locally so the first reset_close_gripper call really publishes a closed hand state.
    right_hand_state_ = hand_open_positions_;
    e3_relative_pose_in_tcp_.orientation.w = 1.0;
    RCLCPP_INFO(
      this->get_logger(),
      "Selected dhz4 arm: arm_side=%s arm_group=%s hand_group=%s tcp_frame=%s ccb_frame=%s d_test_frame=%s zc_target=%s startup_fixed_cache=%s yaw_range=[%.4f, %.4f] yaw_samples=%d",
      arm_side_.c_str(),
      right_arm_group_.c_str(),
      right_hand_group_.c_str(),
      tcp_frame_.c_str(),
      ccb_frame_.c_str(),
      d_test_frame_.c_str(),
      zc_target_name_.c_str(),
      fixed_tail_trajectory_file_.c_str(),
      yaw_sample_range_[0],
      yaw_sample_range_[1],
      yaw_sample_count_);

    plan_client_ = this->create_client<lite_motion_msgs::srv::PlanArmMotion>("plan_arm_motion");
    add_object_client_ = this->create_client<lite_motion_msgs::srv::AddSceneObject>("add_scene_object");
    attach_object_client_ = this->create_client<lite_motion_msgs::srv::AttachSceneObject>("attach_scene_object");
    remove_object_client_ = this->create_client<lite_motion_msgs::srv::RemoveSceneObject>("remove_scene_object");
    set_scene_collision_allowance_client_ = this->create_client<lite_motion_msgs::srv::SetSceneCollisionAllowance>("set_scene_collision_allowance");
    trajectory_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      this->declare_parameter<std::string>(
        "joint_trajectory_topic", "/joint_trajectory_controller/joint_trajectory"),
      10);
    joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
      this->declare_parameter<std::string>("visualization_joint_states_topic", "/joint_states"),
      rclcpp::QoS(10));

    if (enable_visualization_) {
      rclcpp::QoS latched_qos(1);
      latched_qos.transient_local().reliable();
      ref_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "dhz_4/reference_frames", latched_qos);
    }

    start_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      [this]() {
        start_timer_->cancel();
        publishReferenceMarkers();
        worker_ = std::thread([this]() { runSequence(); });
      });
  }

  ~Dhz4Node() override {
    stop_requested_.store(true);
    if (worker_.joinable()) {
      worker_.join();
    }
  }

private:
  struct CachedSegment {
    std::string stage_name;
    std::string group_name;
    trajectory_msgs::msg::JointTrajectory trajectory;
  };

  std::string normalizeArmSide(const std::string& side) const {
    if (side == "0" || side == "left" || side == "l") {
      return "left";
    }
    if (side == "1" || side == "right" || side == "r") {
      return "right";
    }
    throw std::runtime_error("dhz4.arm_side must be left/right or 0/1");
  }

  void selectArm(const std::string& side) {
    if (side == "left") {
      right_arm_group_ = left_arm_group_name_;
      right_hand_group_ = left_hand_group_name_;
      tcp_frame_ = left_tcp_frame_;
      ccb_frame_ = left_ccb_frame_;
      selected_j7_collision_link_ = "arm_left_J7_link";
    } else {
      right_arm_group_ = right_arm_group_name_;
      right_hand_group_ = right_hand_group_name_;
      tcp_frame_ = right_tcp_frame_;
      ccb_frame_ = right_ccb_frame_;
      selected_j7_collision_link_ = "arm_right_J7_link";
    }

  }

  std::vector<std::string> defaultGripperFingerCollisionLinks(const std::string& side) const {
    if (side == "left") {
      return {"zhua_left_1_link", "zhua_left_2_link"};
    }
    return {"zhua_right_1_link", "zhua_right_2_link"};
  }

  std::vector<std::string> allSceneObjectIds() const {
    return {e3_object_id_, zz_object_id_, dhz_object_id_};
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

  void validateZcTargetForArm() const {
    if (arm_side_ == "left" && zc_target_name_ != "zc_m" && zc_target_name_ != "zc_l") {
      throw std::runtime_error("left dhz4 fixed-head playback supports zc_m and zc_l only");
    }
    if (arm_side_ == "right" && zc_target_name_ != "zc_f" && zc_target_name_ != "zc_r") {
      throw std::runtime_error("right dhz4 fixed-head playback supports zc_f and zc_r only");
    }
  }

  std::string makeDefaultFixedHeadTrajectoryFile() const {
    return (std::filesystem::path(fixed_tail_trajectory_dir_) /
      ("dhz4_fixed_head_" + arm_side_ + "_" + zc_target_name_ + ".traj")).string();
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

  bool yawInWrappedRange(double yaw, const std::array<double, 2>& range) const {
    double span = wrapToPi(range[1] - range[0]);
    if (span < 0.0) {
      span += 2.0 * M_PI;
    }
    double rel = wrapToPi(yaw - range[0]);
    if (rel < 0.0) {
      rel += 2.0 * M_PI;
    }
    return rel <= span + 1e-9;
  }

  std::vector<double> sampleYawRange(
    const std::array<double, 2>& range,
    int sample_count) const {
    std::vector<double> samples;
    const int count = std::max(1, sample_count);
    auto append_unique = [&samples](double yaw) {
      const double wrapped = wrapToPi(yaw);
      for (double existing : samples) {
        if (std::abs(wrapToPi(existing - wrapped)) < 1e-6) {
          return;
        }
      }
      samples.push_back(wrapped);
    };
    if (yawInWrappedRange(0.0, range)) {
      append_unique(0.0);
    }
    double span = wrapToPi(range[1] - range[0]);
    if (span < 0.0) {
      span += 2.0 * M_PI;
    }
    if (count == 1) {
      append_unique(range[0] + 0.5 * span);
      return samples;
    }
    for (int i = 0; i < count; ++i) {
      append_unique(range[0] + span * static_cast<double>(i) / static_cast<double>(count - 1));
    }
    return samples;
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

  geometry_msgs::msg::PoseStamped makePoseInDhzFrame(
    const std::array<double, 3>& xyz_dhz,
    const std::array<double, 3>& rpy_dhz) const {
    const pinocchio::SE3 dhz_world = dhzWorldPose();
    const pinocchio::SE3 pose_dhz(
      quaternionEigenFromMsg(
        quaternionMsgFromRpy(rpy_dhz[0], rpy_dhz[1], rpy_dhz[2])).toRotationMatrix(),
      Eigen::Vector3d(xyz_dhz[0], xyz_dhz[1], xyz_dhz[2]));
    geometry_msgs::msg::PoseStamped out;
    out.header.frame_id = planning_frame_;
    out.pose = poseMsgFromSE3(dhz_world * pose_dhz);
    return out;
  }

  geometry_msgs::msg::PoseStamped transformPoseToPlanningFrame(
    const geometry_msgs::msg::PoseStamped& in) {
    if (in.header.frame_id.empty() || in.header.frame_id == planning_frame_) {
      geometry_msgs::msg::PoseStamped out = in;
      out.header.frame_id = planning_frame_;
      return out;
    }

    const pinocchio::SE3 ref_in_planning = currentFramePose(in.header.frame_id);
    const pinocchio::SE3 pose_in_ref = poseMsgToSE3(in.pose);
    geometry_msgs::msg::PoseStamped out;
    out.header.frame_id = planning_frame_;
    out.pose = poseMsgFromSE3(ref_in_planning * pose_in_ref);
    return out;
  }

  geometry_msgs::msg::PoseStamped computeDTestPoseInPlanningFrame() {
    auto d_test_in_ref = makePoseStamped(d_test_xyz_, d_test_rpy_);
    d_test_in_ref.header.frame_id = d_test_frame_;
    return transformPoseToPlanningFrame(d_test_in_ref);
  }

  geometry_msgs::msg::PoseStamped makeDTestPoseInPlanningFrame() {
    if (d_test_world_pose_valid_) {
      return d_test_world_pose_;
    }
    return computeDTestPoseInPlanningFrame();
  }

  geometry_msgs::msg::PoseStamped makeDTestPrePose(
    const geometry_msgs::msg::PoseStamped& d_test_pose) const {
    auto out = offsetAlongLocalAxis(
      d_test_pose,
      Eigen::Vector3d::UnitZ(),
      d_test_pre_offset_along_zz_positive_z_m_);

    Eigen::Matrix3d r_rot;
    r_rot << 0.0, -1.0, 0.0,
             1.0,  0.0, 0.0,
             0.0,  0.0, 1.0;
    const Eigen::Quaterniond base_q = quaternionEigenFromMsg(out.pose.orientation);
    const Eigen::Quaterniond q_new(base_q.toRotationMatrix() * r_rot);

    out.pose.orientation.x = q_new.x();
    out.pose.orientation.y = q_new.y();
    out.pose.orientation.z = q_new.z();
    out.pose.orientation.w = q_new.w();
    return out;
  }

  double posePositionDistance(
    const geometry_msgs::msg::PoseStamped& a,
    const geometry_msgs::msg::PoseStamped& b) const {
    const double dx = a.pose.position.x - b.pose.position.x;
    const double dy = a.pose.position.y - b.pose.position.y;
    const double dz = a.pose.position.z - b.pose.position.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  geometry_msgs::msg::PoseStamped freezeDTestWorldPoseAtCurrentReference(
    const std::string& stage_name) {
    d_test_world_pose_ = computeDTestPoseInPlanningFrame();
    d_test_world_pose_.header.frame_id = planning_frame_;
    d_test_world_pose_valid_ = true;
    logDTestFrameState(stage_name);
    return d_test_world_pose_;
  }

  void logDTestFrameState(const std::string& stage_name) {
    const auto ref_pose = currentFramePoseStamped(d_test_frame_);
    const auto live_d_test_pose = computeDTestPoseInPlanningFrame();
    const auto d_test_pose = makeDTestPoseInPlanningFrame();
    const Eigen::Matrix3d d_test_rot =
      quaternionEigenFromMsg(d_test_pose.pose.orientation).toRotationMatrix();
    const Eigen::Vector3d d_test_z = d_test_rot * Eigen::Vector3d::UnitZ();
    const double live_frozen_delta =
      d_test_world_pose_valid_ ? posePositionDistance(live_d_test_pose, d_test_pose) : 0.0;

    RCLCPP_INFO(
      this->get_logger(),
      "D-test frame [%s]: ref=[%s] ref_in_%s=(%.4f, %.4f, %.4f), configured_ref_xyz=(%.4f, %.4f, %.4f), configured_ref_rpy=(%.4f, %.4f, %.4f), frozen_%s_xyz=(%.4f, %.4f, %.4f), frozen_local_z_in_%s=(%.4f, %.4f, %.4f), live_vs_frozen_delta=%.4f m",
      stage_name.c_str(),
      d_test_frame_.c_str(),
      planning_frame_.c_str(),
      ref_pose.pose.position.x,
      ref_pose.pose.position.y,
      ref_pose.pose.position.z,
      d_test_xyz_[0],
      d_test_xyz_[1],
      d_test_xyz_[2],
      d_test_rpy_[0],
      d_test_rpy_[1],
      d_test_rpy_[2],
      planning_frame_.c_str(),
      d_test_pose.pose.position.x,
      d_test_pose.pose.position.y,
      d_test_pose.pose.position.z,
      planning_frame_.c_str(),
      d_test_z.x(),
      d_test_z.y(),
      d_test_z.z(),
      live_frozen_delta);
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

  Eigen::Vector3d poseLocalAxisInPlanningFrame(
    const geometry_msgs::msg::PoseStamped& pose,
    const Eigen::Vector3d& local_axis) const {
    Eigen::Vector3d axis =
      quaternionEigenFromMsg(pose.pose.orientation).toRotationMatrix() * local_axis.normalized();
    if (axis.norm() < 1e-12) {
      return Eigen::Vector3d::UnitZ();
    }
    return axis.normalized();
  }

  geometry_msgs::msg::PoseStamped rotatePoseAboutLocalZ(
    const geometry_msgs::msg::PoseStamped& in,
    double yaw_offset_rad) const {
    geometry_msgs::msg::PoseStamped out = in;
    const Eigen::Quaterniond base_q = quaternionEigenFromMsg(in.pose.orientation);
    const Eigen::Quaterniond yaw_q(Eigen::AngleAxisd(yaw_offset_rad, Eigen::Vector3d::UnitZ()));
    // Post-multiply so the yaw sample is about the TCP/CCB candidate's own local Z.
    // Pre-multiplying here would make it a planning-frame/base_link yaw.
    const Eigen::Quaterniond q_new = (base_q * yaw_q).normalized();
    out.pose.orientation.x = q_new.x();
    out.pose.orientation.y = q_new.y();
    out.pose.orientation.z = q_new.z();
    out.pose.orientation.w = q_new.w();
    return out;
  }

  pinocchio::SE3 dhzWorldPose() const {
    auto* self = const_cast<Dhz4Node*>(this);
    return poseMsgToSE3(self->makeDTestPoseInPlanningFrame().pose);
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
    return computeArcStepCount(radius_m, angle_rad, arc7_cartesian_max_step_m_, min_step_count_);
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

  Eigen::Matrix3d fixedZzRotationInE3() const {
    return quaternionEigenFromMsg(
      makePoseStamped(
        {0.0, 0.0, 0.0},
        zz_attach_rpy_in_e3_).pose.orientation).toRotationMatrix();
  }

  geometry_msgs::msg::Pose makeZzRelativePoseInTcpUsingLocalAttachRpy() {
    const pinocchio::SE3 tcp_world = currentFramePose(tcp_frame_);
    const pinocchio::SE3 e3_world = tcp_world * poseMsgToSE3(e3_relative_pose_in_tcp_);
    const Eigen::Matrix3d r_world_zz =
      zz_initial_world_pose_valid_ ?
      poseMsgToSE3(zz_initial_world_pose_.pose).rotation() :
      e3_world.rotation() * fixedZzRotationInE3();
    const Eigen::Vector3d p_zz_contact(
      zz_attach_point_xyz_in_zz_[0],
      zz_attach_point_xyz_in_zz_[1],
      zz_attach_point_xyz_in_zz_[2]);

    const Eigen::Vector3d p_world_contact =
      e3_world.translation() +
      e3_world.rotation() * Eigen::Vector3d(
        zz_attach_point_xyz_in_e3_[0],
        zz_attach_point_xyz_in_e3_[1],
        zz_attach_point_xyz_in_e3_[2]);
    const Eigen::Vector3d p_world_zz_origin = p_world_contact - r_world_zz * p_zz_contact;
    const pinocchio::SE3 zz_world(r_world_zz, p_world_zz_origin);
    return poseMsgFromSE3(tcp_world.actInv(zz_world));
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
    const auto zz_relative_to_tcp = makeZzRelativePoseInTcpUsingLocalAttachRpy();
    const auto zz_world_pose = composeFrameRelativePoseStamped(tcp_frame_, zz_relative_to_tcp);

    // Initial E3/zz connection: move zz's configured local contact point onto
    // the current E3 contact point, while preserving zz's original world RPY.
    // This mirrors the later yaw-sweep reattach behavior instead of forcing E3
    // and zz RPY to align.
    if (!addMeshObject(zz_object_id_, zz_mesh_resource_, zz_world_pose)) {
      return false;
    }
    if (!setObjectAttached(zz_object_id_, true, tcp_frame_, &zz_relative_to_tcp)) {
      return false;
    }
    zz_relative_pose_in_tcp_ = zz_relative_to_tcp;
    zz_attached_to_tcp_ = true;
    zz_detached_world_pose_valid_ = false;

    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] fixed zz to attached E3 contact while preserving zz initial world RPY=%s: zz local attach point=(%.4f, %.4f, %.4f) -> E3 contact point=(%.4f, %.4f, %.4f); zz_origin_in_tcp=(%.4f, %.4f, %.4f)",
      stage_name.c_str(),
      zz_initial_world_pose_valid_ ? "true" : "false",
      zz_attach_point_xyz_in_zz_[0], zz_attach_point_xyz_in_zz_[1], zz_attach_point_xyz_in_zz_[2],
      zz_attach_point_xyz_in_e3_[0], zz_attach_point_xyz_in_e3_[1], zz_attach_point_xyz_in_e3_[2],
      zz_relative_to_tcp.position.x, zz_relative_to_tcp.position.y, zz_relative_to_tcp.position.z);
    return true;
  }

  geometry_msgs::msg::Pose makeZzRelativePoseInTcpPreservingDetachedWorldRpy() {
    const pinocchio::SE3 tcp_world = currentFramePose(tcp_frame_);
    const pinocchio::SE3 e3_world = tcp_world * poseMsgToSE3(e3_relative_pose_in_tcp_);
    Eigen::Matrix3d r_world_zz = e3_world.rotation() * fixedZzRotationInE3();
    if (zz_detached_world_pose_valid_) {
      r_world_zz = poseMsgToSE3(zz_detached_world_pose_before_yaw_.pose).rotation();
    }

    const Eigen::Vector3d p_zz_contact(
      zz_attach_point_xyz_in_zz_[0],
      zz_attach_point_xyz_in_zz_[1],
      zz_attach_point_xyz_in_zz_[2]);
    const Eigen::Vector3d p_world_contact =
      e3_world.translation() +
      e3_world.rotation() * Eigen::Vector3d(
        zz_attach_point_xyz_in_e3_[0],
        zz_attach_point_xyz_in_e3_[1],
        zz_attach_point_xyz_in_e3_[2]);
    const Eigen::Vector3d p_world_zz_origin = p_world_contact - r_world_zz * p_zz_contact;
    const pinocchio::SE3 zz_world(r_world_zz, p_world_zz_origin);
    return poseMsgFromSE3(tcp_world.actInv(zz_world));
  }

  bool detachZzToWorldForYawSweep(const std::string& stage_name) {
    // Real scene detach: zz becomes a world object at its current world pose while
    // arm+tcp+E3 yaw samples are evaluated. This avoids rotating zz/tcp through the
    // sampled tolerance and keeps visualization tied to CollisionScene truth.
    if (zz_attached_to_tcp_) {
      zz_detached_world_pose_before_yaw_ =
        composeFrameRelativePoseStamped(tcp_frame_, zz_relative_pose_in_tcp_);
      zz_detached_world_pose_valid_ = true;
    } else {
      zz_detached_world_pose_valid_ = false;
    }
    if (!setObjectAttached(zz_object_id_, false, tcp_frame_)) {
      return false;
    }
    zz_attached_to_tcp_ = false;
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] temporarily detached zz as a world object for yaw sampling; E3 remains rigid on tcp, zz_old_rpy_preserved=%s",
      stage_name.c_str(),
      zz_detached_world_pose_valid_ ? "true" : "false");
    return true;
  }

  bool reattachZzAfterYawSweep(const std::string& stage_name) {
    const auto zz_relative_to_tcp = makeZzRelativePoseInTcpPreservingDetachedWorldRpy();
    const auto zz_world_pose = composeFrameRelativePoseStamped(tcp_frame_, zz_relative_to_tcp);
    if (!addMeshObject(zz_object_id_, zz_mesh_resource_, zz_world_pose)) {
      return false;
    }
    if (!setObjectAttached(zz_object_id_, true, tcp_frame_, &zz_relative_to_tcp)) {
      return false;
    }
    zz_relative_pose_in_tcp_ = zz_relative_to_tcp;
    zz_attached_to_tcp_ = true;
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] reattached zz after yaw sampling: E3 uses the new tcp RPY while zz keeps its pre-yaw world RPY and reconnects at the configured contact point",
      stage_name.c_str());
    zz_detached_world_pose_valid_ = false;
    return true;
  }

  geometry_msgs::msg::PoseStamped offsetCcbAlongDhzLocalAxis(
    const geometry_msgs::msg::PoseStamped& ccb_pose,
    const Eigen::Vector3d& dhz_local_axis,
    double distance_m) {
    geometry_msgs::msg::PoseStamped out = ccb_pose;
    const Eigen::Quaterniond q_dhz = quaternionEigenFromMsg(makeDTestPoseInPlanningFrame().pose.orientation);
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

  bool setSelfCollisionAllowance(double allowed_penetration_m, const std::string& stage_name) {
    return setRuntimeSceneCollisionAllowance(
      std::vector<std::string>{robot_self_collision_allowance_object_id_},
      std::vector<std::string>{"__all_robot_self_collision_pairs__"},
      allowed_penetration_m,
      stage_name);
  }

  bool setBaseGroupCollisionAllowance(double allowed_penetration_m, const std::string& stage_name) {
    return setRuntimeSceneCollisionAllowance(
      std::vector<std::string>{base_group_collision_allowance_object_id_},
      base_group_collision_links_,
      allowed_penetration_m,
      stage_name);
  }

  bool setGripperAndE3CollisionAllowance(double allowed_penetration_m, const std::string& stage_name) {
    if (!setRuntimeSceneCollisionAllowance(
          std::vector<std::string>{e3_object_id_},
          defaultRobotCollisionLinks(),
          allowed_penetration_m,
          stage_name + "_e3_vs_robot")) {
      return false;
    }
    return setGlobalRobotLinkCollisionIgnore(
      defaultGripperFingerCollisionLinks(arm_side_),
      allowed_penetration_m < 0.0,
      stage_name + "_gripper_vs_all");
  }

  bool setFrozenDTestSceneAllowance(double allowed_penetration_m, const std::string& stage_name) {
    const auto links = defaultRobotCollisionLinks();
    if (!setRuntimeSceneCollisionAllowance(
          std::vector<std::string>{zz_object_id_, dhz_object_id_},
          links,
          allowed_penetration_m,
          stage_name + "_zz_dhz_vs_robot")) {
      return false;
    }
    if (!setRuntimeSceneCollisionAllowance(
          std::vector<std::string>{zz_object_id_, dhz_object_id_},
          std::vector<std::string>{e3_object_id_, zz_object_id_, dhz_object_id_},
          allowed_penetration_m,
          stage_name + "_zz_dhz_vs_scene")) {
      return false;
    }
    return true;
  }

  bool setZzGlobalIgnoreAll(double allowed_penetration_m, const std::string& stage_name) {
    std::vector<std::string> links = defaultRobotCollisionLinks();
    const auto scene_ids = allSceneObjectIds();
    links.insert(links.end(), scene_ids.begin(), scene_ids.end());
    return setRuntimeSceneCollisionAllowance(
      std::vector<std::string>{zz_object_id_},
      links,
      allowed_penetration_m,
      stage_name);
  }

  bool setE3SceneAllowance(double allowed_penetration_m, const std::string& stage_name) {
    return setRuntimeSceneCollisionAllowance(
      std::vector<std::string>{e3_object_id_},
      allSceneObjectIds(),
      allowed_penetration_m,
      stage_name);
  }

  bool setDhzAllowance(double allowed_penetration_m, const std::string& stage_name) {
    std::vector<std::string> links = defaultRobotCollisionLinks();
    const auto scene_ids = allSceneObjectIds();
    links.insert(links.end(), scene_ids.begin(), scene_ids.end());
    return setRuntimeSceneCollisionAllowance(
      std::vector<std::string>{dhz_object_id_},
      links,
      allowed_penetration_m,
      stage_name);
  }

  bool setGlobalRobotLinkCollisionIgnore(
    const std::vector<std::string>& link_names,
    bool ignore,
    const std::string& stage_name) {
    if (link_names.empty()) {
      return true;
    }
    return setRuntimeSceneCollisionAllowance(
      std::vector<std::string>{robot_self_collision_ignore_object_id_},
      link_names,
      ignore ? -1.0 : 0.0,
      stage_name);
  }

  bool applyJ7GlobalCollisionIgnore(const std::string& stage_name) {
    return setGlobalRobotLinkCollisionIgnore(
      std::vector<std::string>{selected_j7_collision_link_},
      true,
      stage_name);
  }

  bool clearJ7GlobalCollisionIgnore(const std::string& stage_name) {
    return setGlobalRobotLinkCollisionIgnore(
      std::vector<std::string>{selected_j7_collision_link_},
      false,
      stage_name);
  }

  double effectiveExecutionWaitSec(const trajectory_msgs::msg::JointTrajectory& traj) const {
    double wait_sec = trajectoryDurationSec(traj);
    if (visualization_playback_enabled_) {
      wait_sec /= visualization_playback_speed_scale_;
    }
    return wait_sec + post_action_wait_sec_ + 0.25;
  }

  bool loadFixedTailTrajectory(
    std::vector<CachedSegment>& segments_out,
    std::string& cached_target_out) {
    if (fixed_tail_trajectory_file_.empty()) {
      RCLCPP_ERROR(
        this->get_logger(),
        "dhz4.fixed_tail_trajectory_file must point to the startup fixed trajectory cache.");
      return false;
    }

    std::ifstream in(fixed_tail_trajectory_file_);
    if (!in) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Startup fixed trajectory file not found: %s. Run record_dhz_4_fixed_head for this arm/cache first.",
        fixed_tail_trajectory_file_.c_str());
      return false;
    }

    std::string token;
    in >> token;
    if (token != "lite_motion_fixed_tail_v1") {
      RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file header: %s", token.c_str());
      return false;
    }

    std::string cached_arm_side;
    std::string cached_final_target;
    size_t segment_count = 0;
    in >> token >> cached_arm_side;
    if (token != "arm_side") {
      RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file: expected arm_side.");
      return false;
    }
    in >> token >> cached_final_target;
    if (token != "final_target") {
      RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file: expected final_target.");
      return false;
    }
    if (cached_arm_side != arm_side_) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Fixed-tail cache arm [%s] does not match current dhz4.arm_side [%s].",
        cached_arm_side.c_str(),
        arm_side_.c_str());
      return false;
    }
    in >> token >> segment_count;
    if (token != "segment_count") {
      RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file: expected segment_count.");
      return false;
    }

    std::vector<CachedSegment> parsed;
    parsed.reserve(segment_count);
    for (size_t seg_idx = 0; seg_idx < segment_count; ++seg_idx) {
      CachedSegment segment;
      in >> token;
      if (token != "segment") {
        RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file: expected segment.");
        return false;
      }
      in >> token >> segment.stage_name;
      if (token != "stage") {
        RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file: expected stage.");
        return false;
      }
      in >> token >> segment.group_name;
      if (token != "group") {
        RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file: expected group.");
        return false;
      }

      size_t joint_count = 0;
      in >> token >> joint_count;
      if (token != "joint_names") {
        RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file: expected joint_names.");
        return false;
      }
      segment.trajectory.joint_names.resize(joint_count);
      for (auto& name : segment.trajectory.joint_names) {
        in >> name;
      }

      size_t point_count = 0;
      in >> token >> point_count;
      if (token != "point_count") {
        RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file: expected point_count.");
        return false;
      }
      segment.trajectory.points.reserve(point_count);
      for (size_t point_idx = 0; point_idx < point_count; ++point_idx) {
        trajectory_msgs::msg::JointTrajectoryPoint point;
        double time_sec = 0.0;
        size_t count = 0;
        in >> token >> time_sec;
        if (token != "point") {
          RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file: expected point.");
          return false;
        }
        point.time_from_start = durationFromSec(time_sec);
        in >> token >> count;
        if (token != "positions") {
          RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file: expected positions.");
          return false;
        }
        point.positions.resize(count);
        for (double& value : point.positions) {
          in >> value;
        }
        in >> token >> count;
        if (token != "velocities") {
          RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file: expected velocities.");
          return false;
        }
        point.velocities.resize(count);
        for (double& value : point.velocities) {
          in >> value;
        }
        in >> token >> count;
        if (token != "accelerations") {
          RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file: expected accelerations.");
          return false;
        }
        point.accelerations.resize(count);
        for (double& value : point.accelerations) {
          in >> value;
        }
        segment.trajectory.points.push_back(std::move(point));
      }
      in >> token;
      if (token != "end_segment") {
        RCLCPP_ERROR(this->get_logger(), "Invalid fixed-tail trajectory file: expected end_segment.");
        return false;
      }
      parsed.push_back(std::move(segment));
    }

    if (!in.good() && !in.eof()) {
      RCLCPP_ERROR(this->get_logger(), "Failed while parsing fixed-tail trajectory file.");
      return false;
    }
    segments_out = std::move(parsed);
    cached_target_out = cached_final_target;
    return true;
  }

  bool validateFixedTailSegments(
    const std::vector<CachedSegment>& segments,
    const std::string& cached_target) const {
    const std::string drop_stage = "cartesian_drop_to_" + cached_target;
    const std::string open_stage = "open_gripper_at_" + cached_target;
    return segments.size() == 4 &&
      segments[0].stage_name == drop_stage &&
      segments[0].group_name == right_arm_group_ &&
      segments[1].stage_name == open_stage &&
      segments[1].group_name == right_hand_group_ &&
      segments[2].stage_name == "return_home_after_release_j7_negative_x_retreat" &&
      segments[2].group_name == right_arm_group_ &&
      segments[3].stage_name == "return_home_after_release" &&
      segments[3].group_name == right_arm_group_ &&
      !segments[0].trajectory.points.empty() &&
      !segments[1].trajectory.points.empty() &&
      !segments[2].trajectory.points.empty() &&
      !segments[3].trajectory.points.empty();
  }

  trajectory_msgs::msg::JointTrajectory buildReversedFixedTailPhase(
    const std::vector<CachedSegment>& segments,
    const std::vector<size_t>& reverse_order) const {
    trajectory_msgs::msg::JointTrajectory combined;
    const auto& arm_group = robot_model_->getPlanningGroup(right_arm_group_);
    const auto& hand_group = robot_model_->getPlanningGroup(right_hand_group_);
    combined.joint_names = arm_group.joint_names;
    combined.joint_names.insert(
      combined.joint_names.end(),
      hand_group.joint_names.begin(),
      hand_group.joint_names.end());

    std::unordered_map<std::string, size_t> combined_index;
    combined_index.reserve(combined.joint_names.size());
    for (size_t i = 0; i < combined.joint_names.size(); ++i) {
      combined_index[combined.joint_names[i]] = i;
    }

    std::vector<double> positions(combined.joint_names.size(), 0.0);
    std::vector<double> velocities(combined.joint_names.size(), 0.0);
    std::vector<double> accelerations(combined.joint_names.size(), 0.0);
    for (size_t i = 0; i < arm_group.joint_names.size() && i < right_arm_state_.size(); ++i) {
      positions[combined_index[arm_group.joint_names[i]]] = right_arm_state_[i];
    }
    for (size_t i = 0; i < hand_group.joint_names.size() && i < right_hand_state_.size(); ++i) {
      positions[combined_index[hand_group.joint_names[i]]] = right_hand_state_[i];
    }

    double offset_sec = 0.0;
    double last_output_sec = -1.0;
    for (size_t segment_index : reverse_order) {
      const auto& segment = segments.at(segment_index);
      const double segment_duration = trajectoryDurationSec(segment.trajectory);
      for (auto point_it = segment.trajectory.points.rbegin();
           point_it != segment.trajectory.points.rend();
           ++point_it) {
        std::fill(velocities.begin(), velocities.end(), 0.0);
        std::fill(accelerations.begin(), accelerations.end(), 0.0);
        for (size_t i = 0; i < segment.trajectory.joint_names.size() && i < point_it->positions.size(); ++i) {
          const auto combined_it = combined_index.find(segment.trajectory.joint_names[i]);
          if (combined_it == combined_index.end()) {
            continue;
          }
          const size_t out_index = combined_it->second;
          positions[out_index] = point_it->positions[i];
          if (point_it->velocities.size() == segment.trajectory.joint_names.size()) {
            velocities[out_index] = -point_it->velocities[i];
          }
          if (point_it->accelerations.size() == segment.trajectory.joint_names.size()) {
            accelerations[out_index] = point_it->accelerations[i];
          }
        }

        const double t = offset_sec + segment_duration - pointTimeSec(*point_it);
        if (!combined.points.empty() && std::abs(t - last_output_sec) < 1e-9) {
          combined.points.back().positions = positions;
          combined.points.back().velocities = velocities;
          combined.points.back().accelerations = accelerations;
          continue;
        }

        trajectory_msgs::msg::JointTrajectoryPoint out_point;
        out_point.time_from_start = durationFromSec(t);
        out_point.positions = positions;
        out_point.velocities = velocities;
        out_point.accelerations = accelerations;
        combined.points.push_back(std::move(out_point));
        last_output_sec = t;
      }
      offset_sec += segment_duration;
    }
    return combined;
  }

  void updateStateFromCombinedPoint(
    const trajectory_msgs::msg::JointTrajectory& trajectory,
    const trajectory_msgs::msg::JointTrajectoryPoint& point) {
    std::unordered_map<std::string, size_t> index;
    index.reserve(trajectory.joint_names.size());
    for (size_t i = 0; i < trajectory.joint_names.size(); ++i) {
      index[trajectory.joint_names[i]] = i;
    }

    const auto& arm_group = robot_model_->getPlanningGroup(right_arm_group_);
    for (size_t i = 0; i < arm_group.joint_names.size() && i < right_arm_state_.size(); ++i) {
      const auto it = index.find(arm_group.joint_names[i]);
      if (it != index.end() && it->second < point.positions.size()) {
        right_arm_state_[i] = point.positions[it->second];
      }
    }

    const auto& hand_group = robot_model_->getPlanningGroup(right_hand_group_);
    for (size_t i = 0; i < hand_group.joint_names.size() && i < right_hand_state_.size(); ++i) {
      const auto it = index.find(hand_group.joint_names[i]);
      if (it != index.end() && it->second < point.positions.size()) {
        right_hand_state_[i] = point.positions[it->second];
      }
    }
  }

  sensor_msgs::msg::JointState makeJointStateMsg(
    const Eigen::VectorXd& q_full,
    const rclcpp::Time& stamp) const {
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = stamp;
    msg.name = robot_model_->activeJointNames();
    msg.position.reserve(msg.name.size());
    for (const auto& joint_name : msg.name) {
      msg.position.push_back(q_full[robot_model_->qIndexOfJoint(joint_name)]);
    }
    return msg;
  }

  Eigen::VectorXd makeFullStateFromCombinedPoint(
    const trajectory_msgs::msg::JointTrajectory& trajectory,
    const trajectory_msgs::msg::JointTrajectoryPoint& point) const {
    Eigen::VectorXd q_full = currentFullState();
    const size_t n = std::min(trajectory.joint_names.size(), point.positions.size());
    for (size_t i = 0; i < n; ++i) {
      q_full[robot_model_->qIndexOfJoint(trajectory.joint_names[i])] = point.positions[i];
    }
    return q_full;
  }

  void publishCachedTrajectoryVisualizationPoint(const Eigen::VectorXd& q_full) {
    if (!joint_state_pub_ || !enable_visualization_) {
      return;
    }
    joint_state_pub_->publish(makeJointStateMsg(q_full, this->now()));
  }

  void publishCurrentTheoreticalJointState(const std::string& stage_name) {
    if (!joint_state_pub_ || !enable_visualization_) {
      return;
    }
    joint_state_pub_->publish(makeJointStateMsg(currentFullState(), this->now()));
    RCLCPP_DEBUG(
      this->get_logger(),
      "Stage [%s] published current theoretical full joint state.",
      stage_name.c_str());
  }

  bool publishGroupHoldTrajectory(
    const std::string& group_name,
    const std::vector<double>& target_joints,
    const std::string& stage_name) {
    if (!trajectory_pub_) {
      return true;
    }

    const auto& group = robot_model_->getPlanningGroup(group_name);
    if (target_joints.size() != group.joint_names.size()) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Stage [%s] cannot publish hold trajectory for group [%s]: target size=%zu group dof=%zu",
        stage_name.c_str(),
        group_name.c_str(),
        target_joints.size(),
        group.joint_names.size());
      return false;
    }

    trajectory_msgs::msg::JointTrajectory trajectory;
    trajectory.header.stamp = this->now();
    trajectory.joint_names = group.joint_names;

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = target_joints;
    point.velocities.assign(target_joints.size(), 0.0);
    point.accelerations.assign(target_joints.size(), 0.0);
    point.time_from_start = durationFromSec(std::max(0.02, post_action_wait_sec_));
    trajectory.points.push_back(std::move(point));

    trajectory_pub_->publish(trajectory);
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] published hold trajectory for group [%s] to keep commanded joints at the theoretical target.",
      stage_name.c_str(),
      group_name.c_str());
    return true;
  }

  bool publishCombinedEndpointHoldTrajectory(
    const trajectory_msgs::msg::JointTrajectory& source_trajectory,
    const std::string& stage_name) {
    if (!trajectory_pub_ || source_trajectory.points.empty()) {
      return true;
    }

    const auto& endpoint = source_trajectory.points.back();
    if (endpoint.positions.size() != source_trajectory.joint_names.size()) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Stage [%s] cannot publish combined endpoint hold: joint count=%zu position count=%zu",
        stage_name.c_str(),
        source_trajectory.joint_names.size(),
        endpoint.positions.size());
      return false;
    }

    trajectory_msgs::msg::JointTrajectory hold;
    hold.header.stamp = this->now();
    hold.joint_names = source_trajectory.joint_names;

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = endpoint.positions;
    point.velocities.assign(hold.joint_names.size(), 0.0);
    point.accelerations.assign(hold.joint_names.size(), 0.0);
    point.time_from_start = durationFromSec(std::max(0.02, post_action_wait_sec_));
    hold.points.push_back(std::move(point));

    trajectory_pub_->publish(hold);
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] published endpoint hold trajectory for %zu joint(s).",
      stage_name.c_str(),
      hold.joint_names.size());
    return true;
  }

  bool publishAndTrackCachedTrajectory(
    trajectory_msgs::msg::JointTrajectory trajectory,
    const std::string& stage_name) {
    if (trajectory.points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Cached reverse trajectory phase [%s] is empty.", stage_name.c_str());
      return false;
    }

    trajectory.header.stamp = this->now();
    trajectory_pub_->publish(trajectory);
    RCLCPP_INFO(
      this->get_logger(),
      "Replaying cached fixed trajectory phase [%s] (%zu joints, %zu point(s), %.3f s).",
      stage_name.c_str(),
      trajectory.joint_names.size(),
      trajectory.points.size(),
      trajectoryDurationSec(trajectory));

    updateStateFromCombinedPoint(trajectory, trajectory.points.front());
    publishCachedTrajectoryVisualizationPoint(
      makeFullStateFromCombinedPoint(trajectory, trajectory.points.front()));

    const auto start = std::chrono::steady_clock::now();
    size_t next_idx = 0;
    while (!stop_requested_.load() && next_idx < trajectory.points.size()) {
      const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count() *
        std::max(1e-3, visualization_playback_speed_scale_);
      if (pointTimeSec(trajectory.points[next_idx]) > elapsed) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        continue;
      }
      updateStateFromCombinedPoint(trajectory, trajectory.points[next_idx]);
      publishCachedTrajectoryVisualizationPoint(
        makeFullStateFromCombinedPoint(trajectory, trajectory.points[next_idx]));
      ++next_idx;
    }

    updateStateFromCombinedPoint(trajectory, trajectory.points.back());
    publishCachedTrajectoryVisualizationPoint(
      makeFullStateFromCombinedPoint(trajectory, trajectory.points.back()));
    if (!publishCombinedEndpointHoldTrajectory(trajectory, stage_name + "_endpoint_hold")) {
      return false;
    }
    publishCurrentTheoreticalJointState(stage_name + "_final_sync");
    sleepAfterAction(post_action_wait_sec_);
    return !stop_requested_.load();
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
    bool skip_cartesian_yaw_exact = false,
    int cartesian_min_step_override = -1,
    int cartesian_refinement_max_steps_override = -1,
    std::string* failure_message = nullptr) {
    auto req = std::make_shared<lite_motion_msgs::srv::PlanArmMotion::Request>();
    req->group_name = right_arm_group_;
    req->ee_frame = ee_frame;
    req->use_pose_goal = true;
    req->target_pose = target_pose;
    req->start_joint_positions = right_arm_state_;
    req->plan_only = false;
    req->use_cartesian_path = use_cartesian_path;
    req->cartesian_max_step =
      cartesian_max_step_override_m > 0.0 ? cartesian_max_step_override_m : straight_cartesian_max_step_m_;
    const bool use_yaw_tolerance =
      cartesian_yaw_tolerance_rad > 1e-6 && cartesian_yaw_sample_count > 1;
    if (use_cartesian_path &&
        (cartesian_use_request_solver_settings_ || cartesian_use_sdls_)) {
      req->use_request_solver_settings = true;
      if (cartesian_use_request_solver_settings_) {
        req->request_ik_max_iterations = cartesian_ik_max_iterations_;
        req->request_ik_pos_tolerance = cartesian_ik_pos_tolerance_;
        req->request_ik_rot_tolerance = cartesian_ik_rot_tolerance_;
        req->request_ik_damping = cartesian_ik_damping_;
        req->request_ik_alpha = cartesian_ik_alpha_;
        req->request_ik_max_step_norm = cartesian_ik_max_step_norm_;
        req->request_ik_max_seed_count = cartesian_ik_max_seed_count_;
        req->request_planning_timeout = cartesian_planning_timeout_;
        req->request_path_check_step = cartesian_path_check_step_;
        req->request_global_collision_allowed_penetration = 0.0;
      }
      req->request_cartesian_min_interpolation_steps =
        cartesian_min_step_override > 0 ? cartesian_min_step_override : min_step_count_;
      req->request_cartesian_refinement_max_steps =
        cartesian_refinement_max_steps_override > 0 ?
        cartesian_refinement_max_steps_override :
        refinement_max_steps_;
      req->request_use_sdls = use_cartesian_path && cartesian_use_sdls_;
      req->request_cartesian_nullspace_gain = cartesian_nullspace_gain_;
      req->request_cartesian_singular_threshold = cartesian_singular_threshold_;
      req->request_cartesian_branch_jump_max_rad = cartesian_branch_jump_max_rad_;
      req->request_cartesian_branch_jump_norm_max = cartesian_branch_jump_norm_max_;
      req->request_cartesian_waypoint_pos_tolerance = cartesian_waypoint_pos_tolerance_;
      req->request_cartesian_waypoint_rot_tolerance = cartesian_waypoint_rot_tolerance_;
    }
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
      RCLCPP_INFO(
        this->get_logger(),
        "Stage [%s] enables dhz local +Z yaw sampling: range=[%.4f, %.4f] rad span=%.4f samples=%d skip_exact=%s frame=%s axis_in_%s=(%.4f, %.4f, %.4f).",
        stage_name.c_str(),
        yaw_sample_range_[0],
        yaw_sample_range_[1],
        cartesian_yaw_tolerance_rad,
        cartesian_yaw_sample_count,
        skip_cartesian_yaw_exact ? "true" : "false",
        ee_frame.c_str(),
        planning_frame_.c_str(),
        dhz_z_axis.x(),
        dhz_z_axis.y(),
        dhz_z_axis.z());
    }

    auto resp = callService<lite_motion_msgs::srv::PlanArmMotion>(plan_client_, req, "plan_arm_motion");
    if (!resp || !resp->success) {
      if (failure_message != nullptr) {
        *failure_message = resp ? resp->message : "no response";
      }
      RCLCPP_ERROR(this->get_logger(), "Arm planning failed at stage [%s]: %s", stage_name.c_str(), resp ? resp->message.c_str() : "no response");
      return false;
    }
    if (resp->trajectory.points.empty()) {
      if (failure_message != nullptr) {
        *failure_message = "empty trajectory";
      }
      RCLCPP_ERROR(this->get_logger(), "Arm planning returned an empty trajectory at stage [%s]", stage_name.c_str());
      return false;
    }
    if (failure_message != nullptr) {
      failure_message->clear();
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
    if (resp->used_cartesian_yaw_tolerance) {
      RCLCPP_INFO(
        this->get_logger(),
        "Stage [%s] used dhz-Z yaw sampling; cumulative yaw offset=%.4f rad.",
        stage_name.c_str(),
        resp->cartesian_yaw_offset_rad);
    }

    right_arm_state_ = resp->trajectory.points.back().positions;
    const double wait_sec = effectiveExecutionWaitSec(resp->trajectory);
    RCLCPP_INFO(this->get_logger(), "Stage [%s] trajectory accepted, waiting %.3f s for execution/playback", stage_name.c_str(), wait_sec);
    sleepAfterAction(wait_sec);
    if (!publishGroupHoldTrajectory(right_arm_group_, right_arm_state_, stage_name + "_endpoint_hold")) {
      return false;
    }
    publishCurrentTheoreticalJointState(stage_name + "_final_sync");
    return true;
  }

  std::shared_ptr<lite_motion_msgs::srv::PlanArmMotion::Response> planArmPoseOnly(
    const geometry_msgs::msg::PoseStamped& target_pose,
    const std::string& ee_frame,
    bool use_cartesian_path,
    const std::string& stage_name,
    double cartesian_max_step_override_m = -1.0,
    int cartesian_min_step_override = -1,
    int cartesian_refinement_max_steps_override = -1,
    bool use_d2_relaxed_settings = false) {
    auto req = std::make_shared<lite_motion_msgs::srv::PlanArmMotion::Request>();
    req->group_name = right_arm_group_;
    req->ee_frame = ee_frame;
    req->use_pose_goal = true;
    req->target_pose = target_pose;
    req->start_joint_positions = right_arm_state_;
    req->plan_only = true;
    req->use_cartesian_path = use_cartesian_path;
    req->cartesian_max_step =
      cartesian_max_step_override_m > 0.0 ? cartesian_max_step_override_m : straight_cartesian_max_step_m_;
    if (use_d2_relaxed_settings && !use_cartesian_path) {
      req->use_weighted_orientation_goal = true;
      req->orientation_weight_x = 1.0;
      req->orientation_weight_y = 1.0;
      req->orientation_weight_z = 1.0;
    }
    if (use_d2_relaxed_settings || (use_cartesian_path && cartesian_use_request_solver_settings_)) {
      req->use_request_solver_settings = true;
      req->request_ik_max_iterations =
        use_d2_relaxed_settings ? arc7_d2_ik_max_iterations_ : cartesian_ik_max_iterations_;
      req->request_ik_pos_tolerance =
        use_d2_relaxed_settings ? arc7_d2_ik_pos_tolerance_ : cartesian_ik_pos_tolerance_;
      req->request_ik_rot_tolerance =
        use_d2_relaxed_settings ? arc7_d2_ik_rot_tolerance_ : cartesian_ik_rot_tolerance_;
      req->request_ik_damping =
        use_d2_relaxed_settings ? arc7_d2_ik_damping_ : cartesian_ik_damping_;
      req->request_ik_alpha =
        use_d2_relaxed_settings ? arc7_d2_ik_alpha_ : cartesian_ik_alpha_;
      req->request_ik_max_step_norm =
        use_d2_relaxed_settings ? arc7_d2_ik_max_step_norm_ : cartesian_ik_max_step_norm_;
      req->request_ik_max_seed_count =
        use_d2_relaxed_settings ? arc7_d2_ik_max_seed_count_ : cartesian_ik_max_seed_count_;
      req->request_max_ik_goal_candidates =
        use_d2_relaxed_settings && !use_cartesian_path ? arc7_d2_max_ik_goal_candidates_ : 0;
      req->request_planning_timeout =
        use_d2_relaxed_settings ? arc7_d2_planning_timeout_ : cartesian_planning_timeout_;
      req->request_path_check_step =
        use_d2_relaxed_settings ? arc7_d2_path_check_step_ : cartesian_path_check_step_;
      req->request_cartesian_min_interpolation_steps =
        cartesian_min_step_override > 0 ? cartesian_min_step_override : min_step_count_;
      req->request_cartesian_refinement_max_steps =
        cartesian_refinement_max_steps_override > 0 ?
        cartesian_refinement_max_steps_override :
        refinement_max_steps_;
      req->request_use_sdls = false;
      req->request_cartesian_nullspace_gain = cartesian_nullspace_gain_;
      req->request_cartesian_singular_threshold = cartesian_singular_threshold_;
      req->request_cartesian_branch_jump_max_rad = cartesian_branch_jump_max_rad_;
      req->request_cartesian_branch_jump_norm_max = cartesian_branch_jump_norm_max_;
      req->request_cartesian_waypoint_pos_tolerance =
        use_d2_relaxed_settings ? arc7_d2_waypoint_pos_tolerance_ : cartesian_waypoint_pos_tolerance_;
      req->request_cartesian_waypoint_rot_tolerance =
        use_d2_relaxed_settings ? arc7_d2_waypoint_rot_tolerance_ : cartesian_waypoint_rot_tolerance_;
      if (use_d2_relaxed_settings) {
        RCLCPP_INFO(
          this->get_logger(),
          "Plan-only stage [%s] uses relaxed D_2 solver settings: ik_iter=%d pos_tol=%.4f rot_tol=%.4f step=%.3f seeds=%d max_ik_goals=%d timeout=%.2f path_step=%.3f cart_wp_tol=(%.4f, %.4f), sdls=off.",
          stage_name.c_str(),
          arc7_d2_ik_max_iterations_,
          arc7_d2_ik_pos_tolerance_,
          arc7_d2_ik_rot_tolerance_,
          arc7_d2_ik_max_step_norm_,
          arc7_d2_ik_max_seed_count_,
          req->request_max_ik_goal_candidates,
          arc7_d2_planning_timeout_,
          arc7_d2_path_check_step_,
          arc7_d2_waypoint_pos_tolerance_,
          arc7_d2_waypoint_rot_tolerance_);
      }
    }
    auto resp = callService<lite_motion_msgs::srv::PlanArmMotion>(
      plan_client_,
      req,
      "plan_arm_motion");
    if (!resp || !resp->success) {
      RCLCPP_WARN(
        this->get_logger(),
        "Plan-only arm pose failed at stage [%s]: %s",
        stage_name.c_str(),
        resp ? resp->message.c_str() : "no response");
    }
    return resp;
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
    bool allow_zz_detach_for_yaw = false,
    int cartesian_min_step_override = -1,
    int cartesian_refinement_max_steps_override = -1) {
    const bool use_yaw_tolerance = yaw_tolerance_rad > 1e-6 && yaw_sample_count > 1;
    const bool can_detach_zz_for_yaw =
      use_cartesian_path &&
      allow_zz_detach_for_yaw &&
      detach_zz_during_attached_chain_yaw_sweep_ &&
      use_yaw_tolerance &&
      zz_attached_to_tcp_;
    auto is_non_singular_joint_limit_failure =
      [](const std::string& message) {
        return message.find("joint_limit=") != std::string::npos &&
               message.find("near_singular=false") != std::string::npos;
      };

    if (can_detach_zz_for_yaw) {
      std::string attached_failure;
      const auto attempt_with_current_attachment =
        [this,
         &target_pose,
         &ee_frame,
         use_cartesian_path,
         &stage_name,
         cartesian_max_step_override_m,
         cartesian_min_step_override,
         cartesian_refinement_max_steps_override,
         executed_pose,
         &attached_failure]() {
          if (executeArmPosePlan(
                target_pose,
                ee_frame,
                use_cartesian_path,
                stage_name,
                cartesian_max_step_override_m,
                0.0,
                1,
                executed_pose,
                false,
                cartesian_min_step_override,
                cartesian_refinement_max_steps_override,
                &attached_failure)) {
            return true;
          }
          return false;
        };

      if (attempt_with_current_attachment()) {
        return true;
      }
      if (is_non_singular_joint_limit_failure(attached_failure)) {
        RCLCPP_WARN(
          this->get_logger(),
          "Stage [%s] will not run detached/yaw retry after attached exact failure because the blocker is a non-singular joint-limit/branch issue: %s",
          stage_name.c_str(),
          attached_failure.c_str());
        return false;
      }

      if (!detachZzToWorldForYawSweep(stage_name + "_before_cartesian_yaw_tolerance")) {
        return false;
      }

      bool zz_detached_for_yaw = true;
      std::string detached_failure;
      bool plan_ok = executeArmPosePlan(
        target_pose,
        ee_frame,
        use_cartesian_path,
        stage_name + "_detached_exact",
        cartesian_max_step_override_m,
        0.0,
        1,
        executed_pose,
        false,
        cartesian_min_step_override,
        cartesian_refinement_max_steps_override,
        &detached_failure);
      if (!plan_ok && !is_non_singular_joint_limit_failure(detached_failure)) {
        plan_ok = executeArmPosePlan(
          target_pose,
          ee_frame,
          use_cartesian_path,
          stage_name,
          cartesian_max_step_override_m,
          yaw_tolerance_rad,
          yaw_sample_count,
          executed_pose,
          false,
          cartesian_min_step_override,
          cartesian_refinement_max_steps_override);
      } else if (!plan_ok) {
        RCLCPP_WARN(
          this->get_logger(),
          "Stage [%s] skipped yaw sampling after detached exact failure because the blocker is still a non-singular joint-limit/branch issue: %s",
          stage_name.c_str(),
          detached_failure.c_str());
      }
      if (plan_ok) {
        if (!reattachZzAfterYawSweep(stage_name + "_reattach_after_success")) {
          return false;
        }
        return true;
      }

      if (zz_detached_for_yaw) {
        (void)reattachZzAfterYawSweep(stage_name + "_reattach_after_failed_sweep");
      }
      return false;
    }

    bool plan_ok = executeArmPosePlan(
      target_pose,
      ee_frame,
      use_cartesian_path,
      stage_name,
      cartesian_max_step_override_m,
      yaw_tolerance_rad,
      yaw_sample_count,
      executed_pose,
      false,
      cartesian_min_step_override,
      cartesian_refinement_max_steps_override);
    if (plan_ok) {
      return true;
    }

    return false;
  }

  bool executeCcbPosePlanWithYawSweep(
    const geometry_msgs::msg::PoseStamped& target_pose,
    bool use_cartesian_path,
    const std::string& stage_name,
    geometry_msgs::msg::PoseStamped* executed_pose = nullptr,
    double yaw_tolerance_rad = 0.0,
    int yaw_sample_count = 1,
    double cartesian_max_step_override_m = -1.0,
    int cartesian_min_step_override = -1,
    int cartesian_refinement_max_steps_override = -1) {
    if (yaw_tolerance_rad > 1e-6 && yaw_sample_count > 1) {
      if (executeArmPosePlan(
            target_pose,
            ccb_frame_,
            use_cartesian_path,
            stage_name,
            cartesian_max_step_override_m,
            0.0,
            1,
            executed_pose,
            false,
            cartesian_min_step_override,
            cartesian_refinement_max_steps_override)) {
        return true;
      }
    }
    if (executeArmPosePlan(
          target_pose,
          ccb_frame_,
          use_cartesian_path,
          stage_name,
          cartesian_max_step_override_m,
          yaw_tolerance_rad,
          yaw_sample_count,
          executed_pose,
          !use_cartesian_path && yaw_tolerance_rad > 1e-6 && yaw_sample_count > 1,
          cartesian_min_step_override,
          cartesian_refinement_max_steps_override)) {
      return true;
    }
    return false;
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
    bool allow_zz_detach_for_yaw = false,
    int refinement_max_steps = -1) {
    auto rolling_pose = currentFramePoseStamped(ccb_frame_);
    const Eigen::Vector3d center_world = dhzLocalPointToWorld(center_xyz_dhz);
    const Eigen::Vector3d axis_world = dhzLocalAxisToWorld(axis_dhz);
    const Eigen::Vector3d start_pos(
      rolling_pose.pose.position.x,
      rolling_pose.pose.position.y,
      rolling_pose.pose.position.z);
    const double radius_m = (start_pos - center_world).norm();
    const double arc_step_m = cartesian_max_step_m > 0.0 ? cartesian_max_step_m : arc7_cartesian_max_step_m_;
    const int arc_min_steps = min_step_count > 0 ? min_step_count : min_step_count_;
    const int arc_refinement_steps =
      refinement_max_steps > 0 ? refinement_max_steps : refinement_max_steps_;
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
            allow_zz_detach_for_yaw,
            arc_min_steps,
            arc_refinement_steps)) {
        return false;
      }
      rolling_pose = executed_pose;
    }
    return true;
  }

  bool planOnlyCcbArcAroundDhz(
    const std::array<double, 3>& center_xyz_dhz,
    const Eigen::Vector3d& axis_dhz,
    double angle_rad,
    const std::string& stage_name,
    double cartesian_max_step_m,
    int min_step_count,
    int refinement_max_steps) {
    auto rolling_pose = currentFramePoseStamped(ccb_frame_);
    const Eigen::Vector3d center_world = dhzLocalPointToWorld(center_xyz_dhz);
    const Eigen::Vector3d axis_world = dhzLocalAxisToWorld(axis_dhz);
    const Eigen::Vector3d start_pos(
      rolling_pose.pose.position.x,
      rolling_pose.pose.position.y,
      rolling_pose.pose.position.z);
    const double radius_m = (start_pos - center_world).norm();
    const double arc_step_m = cartesian_max_step_m > 0.0 ? cartesian_max_step_m : arc7_d2_cartesian_max_step_m_;
    const int arc_min_steps = min_step_count > 0 ? min_step_count : arc7_d2_min_step_count_;
    const int arc_refinement_steps =
      refinement_max_steps > 0 ? refinement_max_steps : arc7_d2_refinement_max_steps_;
    const size_t step_count = computeArcStepCount(radius_m, angle_rad, arc_step_m, arc_min_steps);

    RCLCPP_INFO(
      this->get_logger(),
      "Plan-only stage [%s] dry-runs complete arc around dhz: center=(%.4f, %.4f, %.4f) radius=%.5f angle=%.4f rad outer_steps=%zu max_step=%.4f min_steps=%d refinement_max=%d yaw_sampling=off sdls=off.",
      stage_name.c_str(),
      center_world.x(), center_world.y(), center_world.z(),
      radius_m,
      angle_rad,
      step_count,
      arc_step_m,
      arc_min_steps,
      arc_refinement_steps);

    for (size_t i = 1; i <= step_count; ++i) {
      const double theta_step = angle_rad / static_cast<double>(step_count);
      const auto waypoint =
        rotatePoseAboutWorldAxis(rolling_pose, center_world, axis_world, theta_step);
      const std::string waypoint_stage =
        stage_name + "_wp_" + std::to_string(i) + "_of_" + std::to_string(step_count);
      auto resp = planArmPoseOnly(
        waypoint,
        ccb_frame_,
        true,
        waypoint_stage,
        arc_step_m,
        arc_min_steps,
        arc_refinement_steps,
        true);
      if (!resp || !resp->success || resp->trajectory.points.empty()) {
        RCLCPP_WARN(
          this->get_logger(),
          "Plan-only complete arc dry-run failed at [%s]: %s",
          waypoint_stage.c_str(),
          resp ? resp->message.c_str() : "no response");
        return false;
      }
      right_arm_state_ = resp->trajectory.points.back().positions;
      rolling_pose = waypoint;
    }
    return true;
  }

  struct D2PreSolveResult {
    bool found{false};
    std::vector<double> d2_joints;
    double yaw_rad{0.0};
  };

  D2PreSolveResult preSolveArc7D2() {
    D2PreSolveResult result;
    if (!arc7_pre_solve_d2_enabled_) {
      return result;
    }

    const auto original_arm_state = right_arm_state_;
    const auto original_hand_state = right_hand_state_;
    const bool original_zz_attached = zz_attached_to_tcp_;
    const bool original_zz_detached_valid = zz_detached_world_pose_valid_;
    const auto original_zz_relative = zz_relative_pose_in_tcp_;
    const auto original_zz_detached_pose = zz_detached_world_pose_before_yaw_;

    auto restore_state = [&]() {
      right_arm_state_ = original_arm_state;
      right_hand_state_ = original_hand_state;
      zz_attached_to_tcp_ = original_zz_attached;
      zz_detached_world_pose_valid_ = original_zz_detached_valid;
      zz_relative_pose_in_tcp_ = original_zz_relative;
      zz_detached_world_pose_before_yaw_ = original_zz_detached_pose;
    };

    const auto yaw_samples = sampleYawRange(
      arc7_d2_yaw_sample_range_,
      arc7_d2_yaw_sample_count_);
    RCLCPP_INFO(
      this->get_logger(),
      "Pre-solving arc7 D_2 from dhz-local ccb start xyz=(%.4f, %.4f, %.4f) rpy=(%.4f, %.4f, %.4f) yaw_range=[%.4f, %.4f] samples=%zu; dry-run arc7 max_step=%.4f min_steps=%d refinement_max=%d.",
      arc7_d2_start_xyz_dhz_[0],
      arc7_d2_start_xyz_dhz_[1],
      arc7_d2_start_xyz_dhz_[2],
      arc7_d2_start_rpy_dhz_[0],
      arc7_d2_start_rpy_dhz_[1],
      arc7_d2_start_rpy_dhz_[2],
      arc7_d2_yaw_sample_range_[0],
      arc7_d2_yaw_sample_range_[1],
      yaw_samples.size(),
      arc7_d2_cartesian_max_step_m_,
      arc7_d2_min_step_count_,
      arc7_d2_refinement_max_steps_);

    if (!setDhzAllowance(
          arc7_dhz_allowed_penetration_m_,
          "arc7_pre_solve_temp_allow_dhz_0p006")) {
      restore_state();
      return result;
    }
    auto restore_dhz_allowance = [&]() {
      (void)setDhzAllowance(
        frozen_d_test_scene_allowed_penetration_m_,
        "arc7_pre_solve_restore_dhz_0p002");
    };

    for (size_t i = 0; i < yaw_samples.size(); ++i) {
      auto candidate_rpy = arc7_d2_start_rpy_dhz_;
      candidate_rpy[2] += yaw_samples[i];
      const auto d2_pose = makePoseInDhzFrame(arc7_d2_start_xyz_dhz_, candidate_rpy);
      restore_state();
      const std::string d2_stage =
        "arc7_pre_solve_D2_yaw_" + std::to_string(i + 1) + "_of_" + std::to_string(yaw_samples.size());
      auto d2_resp = planArmPoseOnly(
            d2_pose,
            ccb_frame_,
            false,
            d2_stage,
            -1.0,
            -1,
            -1,
            true);
      if (!d2_resp || !d2_resp->success || d2_resp->trajectory.points.empty()) {
        RCLCPP_WARN(
          this->get_logger(),
          "Pre-solve D_2 yaw candidate %zu/%zu yaw=%.4f failed to reach D_2 pose: %s",
          i + 1,
          yaw_samples.size(),
          yaw_samples[i],
          d2_resp ? d2_resp->message.c_str() : "no response");
        continue;
      }

      const auto d2_joints = d2_resp->trajectory.points.back().positions;
      right_arm_state_ = d2_joints;
      const bool arc7_ok = planOnlyCcbArcAroundDhz(
        arc7_center_xyz_dhz_,
        Eigen::Vector3d::UnitX(),
        arc7_roll_angle_rad_,
        "arc7_pre_solve_dry_run_from_D2_yaw_" + std::to_string(i + 1),
        arc7_d2_cartesian_max_step_m_,
        arc7_d2_min_step_count_,
        arc7_d2_refinement_max_steps_);
      if (!arc7_ok) {
        RCLCPP_WARN(
          this->get_logger(),
          "Pre-solve D_2 yaw candidate %zu/%zu yaw=%.4f reached D_2, but complete dry-run arc7 failed.",
          i + 1,
          yaw_samples.size(),
          yaw_samples[i]);
        restore_state();
        continue;
      }

      result.found = true;
      result.d2_joints = d2_joints;
      result.yaw_rad = yaw_samples[i];
      RCLCPP_INFO(
        this->get_logger(),
        "Pre-solve D_2 succeeded with yaw candidate %zu/%zu yaw=%.4f. D_2 joints=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f].",
        i + 1,
        yaw_samples.size(),
        yaw_samples[i],
        d2_joints.size() > 0 ? d2_joints[0] : 0.0,
        d2_joints.size() > 1 ? d2_joints[1] : 0.0,
        d2_joints.size() > 2 ? d2_joints[2] : 0.0,
        d2_joints.size() > 3 ? d2_joints[3] : 0.0,
        d2_joints.size() > 4 ? d2_joints[4] : 0.0,
        d2_joints.size() > 5 ? d2_joints[5] : 0.0,
        d2_joints.size() > 6 ? d2_joints[6] : 0.0);
      restore_state();
      restore_dhz_allowance();
      return result;
    }

    restore_state();
    restore_dhz_allowance();
    RCLCPP_ERROR(
      this->get_logger(),
      "Pre-solve D_2 failed: no yaw candidate could reach D_2 and complete a dry-run arc7.");
    return result;
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
        right_arm_state_ = target_joints;
        if (!publishGroupHoldTrajectory(right_arm_group_, target_joints, stage_name)) {
          return false;
        }
        publishCurrentTheoreticalJointState(stage_name + "_skip_sync");
        sleepAfterAction(post_action_wait_sec_);
        return true;
      }
    }

    auto req = std::make_shared<lite_motion_msgs::srv::PlanArmMotion::Request>();
    req->group_name = right_arm_group_;
    req->ee_frame = ccb_frame_;
    req->use_pose_goal = false;
    req->start_joint_positions = right_arm_state_;
    req->goal_joint_positions = target_joints;
    req->plan_only = false;
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
    if (!publishGroupHoldTrajectory(right_arm_group_, right_arm_state_, stage_name + "_endpoint_hold")) {
      return false;
    }
    publishCurrentTheoreticalJointState(stage_name + "_final_sync");
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
        right_hand_state_ = target_joints;
        if (!publishGroupHoldTrajectory(right_hand_group_, target_joints, stage_name)) {
          return false;
        }
        publishCurrentTheoreticalJointState(stage_name + "_skip_sync");
        sleepAfterAction(post_action_wait_sec_);
        return true;
      }
    }

    auto req = std::make_shared<lite_motion_msgs::srv::PlanArmMotion::Request>();
    req->group_name = right_hand_group_;
    req->ee_frame = tcp_frame_;
    req->use_pose_goal = false;
    req->start_joint_positions = right_hand_state_;
    req->goal_joint_positions = target_joints;
    req->plan_only = false;
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
    if (!publishGroupHoldTrajectory(right_hand_group_, right_hand_state_, stage_name + "_endpoint_hold")) {
      return false;
    }
    publishCurrentTheoreticalJointState(stage_name + "_final_sync");
    return true;
  }

  bool createAndAttachE3AtCurrentTcp(const std::string& stage_name) {
    const auto tcp_pose = currentFramePoseStamped(tcp_frame_);
    if (!addMeshObject(e3_object_id_, e3_mesh_resource_, tcp_pose)) {
      return false;
    }
    e3_relative_pose_in_tcp_ = geometry_msgs::msg::Pose();
    e3_relative_pose_in_tcp_.orientation.w = 1.0;
    if (!setObjectAttached(e3_object_id_, true, tcp_frame_, &e3_relative_pose_in_tcp_)) {
      return false;
    }
    sleepAfterAction(post_action_wait_sec_);
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] created E3 at current [%s] and attached its visual/collision model after the startup fixed trajectory endpoint was reached.",
      stage_name.c_str(),
      tcp_frame_.c_str());
    return true;
  }

  bool initializeSceneAtHome() {
    // dhz_4 starts without E3 in the scene.  E3 is created only after the startup
    // fixed trajectory reaches its endpoint, so the scene matches the requested timing.
    e3_relative_pose_in_tcp_ = geometry_msgs::msg::Pose();
    e3_relative_pose_in_tcp_.orientation.w = 1.0;
    zz_relative_pose_in_tcp_ = geometry_msgs::msg::Pose();
    zz_relative_pose_in_tcp_.orientation.w = 1.0;
    zz_initial_world_pose_ = geometry_msgs::msg::PoseStamped();
    zz_initial_world_pose_.pose.orientation.w = 1.0;
    zz_attached_to_tcp_ = false;
    zz_detached_world_pose_valid_ = false;
    zz_initial_world_pose_valid_ = false;
    d_test_world_pose_valid_ = false;
    return true;
  }

  bool moveToObservePoseAfterStartupPlayback() {
    const std::string stage_prefix = "move_to_" + arm_side_ + "_observe_after_startup_playback";
    if (!executeArmJointPlan(observe_arm_positions_, stage_prefix + "_arm")) {
      return false;
    }
    if (!executeHandJointPlan(observe_hand_positions_, stage_prefix + "_hand")) {
      return false;
    }
    RCLCPP_INFO(
      this->get_logger(),
      "Reached [%s] observe pose after startup cached playback; next step will spawn zz/dhz at d_test in [%s].",
      arm_side_.c_str(),
      d_test_frame_.c_str());
    return true;
  }

  bool addZzAndDhzAtDTestAfterObserve() {
    const auto d_test_pose = makeDTestPoseInPlanningFrame();
    // zz and dhz are inserted only after startup playback and observe pose are complete.
    // Their configured pose is interpreted in d_test_frame_ (camera link) then converted.
    if (!addMeshObject(zz_object_id_, zz_mesh_resource_, d_test_pose)) {
      return false;
    }
    zz_initial_world_pose_ = d_test_pose;
    zz_initial_world_pose_valid_ = true;
    if (!addMeshObject(dhz_object_id_, dhz_mesh_resource_, d_test_pose)) {
      return false;
    }
    RCLCPP_INFO(
      this->get_logger(),
      "Spawned zz/dhz at frozen d_test (reference frame [%s]) after observe pose: %s_xyz=(%.4f, %.4f, %.4f).",
      d_test_frame_.c_str(),
      planning_frame_.c_str(),
      d_test_pose.pose.position.x,
      d_test_pose.pose.position.y,
      d_test_pose.pose.position.z);
    return true;
  }

  bool playStartupFixedTrajectoryAndAttachE3() {
    std::vector<CachedSegment> segments;
    std::string cached_target;
    if (!loadFixedTailTrajectory(segments, cached_target)) {
      return false;
    }
    if (cached_target != zc_target_name_) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Startup fixed trajectory cache target [%s] does not match dhz4.zc_target [%s].",
        cached_target.c_str(),
        zc_target_name_.c_str());
      return false;
    }
    if (!validateFixedTailSegments(segments, cached_target)) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Startup fixed trajectory cache [%s] must contain exactly drop/open/J7-retreat/return_home_after_release for arm [%s] and target [%s].",
        fixed_tail_trajectory_file_.c_str(),
        arm_side_.c_str(),
        cached_target.c_str());
      return false;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Using startup fixed trajectory cache [%s] to reach cache endpoint [%s] with no E3 in the scene; internal state is advanced before downstream planning.",
      fixed_tail_trajectory_file_.c_str(),
      cached_target.c_str());

    auto startup_endpoint_trajectory = buildReversedFixedTailPhase(segments, {3, 2, 1, 0});
    if (!publishAndTrackCachedTrajectory(
          std::move(startup_endpoint_trajectory),
          "startup_fixed_trajectory_to_" + cached_target + "_endpoint")) {
      return false;
    }

    if (!createAndAttachE3AtCurrentTcp("after_startup_fixed_trajectory_" + cached_target)) {
      return false;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "E3 was created at [%s] after startup fixed trajectory endpoint [%s] and attached to [%s]; downstream zz/dhz motion keeps the remaining dhz_4 flow.",
      tcp_frame_.c_str(),
      cached_target.c_str(),
      tcp_frame_.c_str());
    return true;
  }

  bool releaseObjectsAndReturnHome() {
    // Release the gripper first, then break the visual/collision attachment relation.
    if (!executeHandJointPlan(hand_open_positions_, "open_gripper_after_dhz4_finish")) {
      return false;
    }

    if (!setGripperAndE3CollisionAllowance(
          0.0,
          "after_open_gripper_restore_gripper_collision")) {
      return false;
    }
    if (!setE3SceneAllowance(
          return_home_e3_allowed_penetration_m_,
          "after_open_gripper_allow_e3_scene_0p002")) {
      return false;
    }
    if (!setSelfCollisionAllowance(
          return_home_self_collision_allowed_penetration_m_,
          "after_open_gripper_set_self_collision_0p001")) {
      return false;
    }

    // zz is attached during the legacy downstream flow.  Detach it before E3 so the
    // robot does not carry either scene object while returning to the all-zero posture.
    (void)setObjectAttached(zz_object_id_, false, tcp_frame_);
    zz_attached_to_tcp_ = false;
    zz_detached_world_pose_valid_ = false;
    if (!setObjectAttached(e3_object_id_, false, tcp_frame_)) {
      return false;
    }

    removeObject(e3_object_id_, false);
    removeObject(zz_object_id_, false);

    const bool home_ok = executeArmJointPlan(arm_home_positions_, "return_home_after_release", true);

    return home_ok;
  }

  bool runSequenceOnce() {
    if (!initializeSceneAtHome()) {
      return false;
    }

    if (!setBaseGroupCollisionAllowance(
          base_group_collision_allowed_penetration_m_,
          "initialize_base_group_allowance_0p010_until_cleanup")) {
      return false;
    }

    if (!setGripperAndE3CollisionAllowance(
          gripper_e3_ignore_allowed_penetration_m_,
          "initialize_ignore_gripper_and_e3_until_return_home")) {
      return false;
    }

    if (!playStartupFixedTrajectoryAndAttachE3()) {
      return false;
    }

    if (!applyJ7GlobalCollisionIgnore("before_observe_pose_ignore_j7_all_collisions")) {
      return false;
    }

    if (!moveToObservePoseAfterStartupPlayback()) {
      return false;
    }

    if (!clearJ7GlobalCollisionIgnore("after_observe_pose_restore_j7_collisions")) {
      return false;
    }

    const auto d_test_pose = freezeDTestWorldPoseAtCurrentReference("after_observe_pose");
    const auto d_test_pre_pose = makeDTestPrePose(d_test_pose);
    geometry_msgs::msg::PoseStamped reached_ccb_pose = d_test_pre_pose;
    publishReferenceMarkers("after_observe_pose_d_test_frozen");

    if (!addZzAndDhzAtDTestAfterObserve()) {
      return false;
    }
    publishReferenceMarkers("after_spawn_zz_dhz_at_frozen_d_test");

    if (!setFrozenDTestSceneAllowance(
          frozen_d_test_scene_allowed_penetration_m_,
          "after_spawn_zz_dhz_allow_zz_dhz_0p002_until_finish")) {
      return false;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Planning %s to d_test pre-offset along dhz local +Z by %.4f m; IK fallback may use dhz-Z yaw sampling in range [%.4f, %.4f].",
      ccb_frame_.c_str(),
      d_test_pre_offset_along_zz_positive_z_m_,
      yaw_sample_range_[0],
      yaw_sample_range_[1]);
    if (!executeCcbPosePlanWithYawSweep(
          d_test_pre_pose,
          false,
          "move_ccb_to_d_test_zz_positive_z_pre_offset",
          &reached_ccb_pose,
          yawSampleSpan(),
          yaw_sample_count_)) {
      return false;
    }
    publishReferenceMarkers("after_move_ccb_to_frozen_d_test_pre_pose");

    if (!setZzGlobalIgnoreAll(
          -1.0,
          "after_move_to_d_test_pre_pose_ignore_zz_all_collisions")) {
      return false;
    }

    // At this point only E3 is attached to TCP. Move along negative Z first; zz
    // remains the independent world object at the frozen d_test until this move finishes.
    const auto down_pose = offsetAlongLocalAxis(reached_ccb_pose, Eigen::Vector3d::UnitZ(), -ccb_approach_distance_m_);
    if (!executeArmPosePlanWithLocalYawSweep(
          down_pose,
          ccb_frame_,
          true,
          "move_ccb_along_negative_z_0p113_before_zz_e3_attach",
          straight_cartesian_max_step_m_,
          yawSampleSpan(),
          yaw_sample_count_,
          nullptr,
          false,
          min_step_count_,
          refinement_max_steps_)) {
      return false;
    }
    publishReferenceMarkers("after_move_ccb_along_negative_z_before_zz_attach");

    // Establish the E3/zz connection immediately after the negative-Z approach,
    // before the +Y motion starts.
    if (!attachZzToTcpWithE3ContactPoint("after_negative_z_0p113_connect_zz_to_e3")) {
      return false;
    }
    publishReferenceMarkers("after_negative_z_0p113_zz_attached");

    const D2PreSolveResult d2_pre_solve = preSolveArc7D2();
    if (arc7_pre_solve_d2_enabled_ && !d2_pre_solve.found) {
      return false;
    }

    const auto ccb_after_down_pose = currentFramePoseStamped(ccb_frame_);
    const Eigen::Vector3d dhz_positive_y_world = dhzLocalAxisToWorld(Eigen::Vector3d::UnitY());
    if (d2_pre_solve.found) {
      RCLCPP_INFO(
        this->get_logger(),
        "Replacing +Y Cartesian push endpoint with pre-solved D_2 joints from yaw %.4f, so arc7 starts from a known feasible joint branch.",
        d2_pre_solve.yaw_rad);
      if (!executeArmJointPlan(
            d2_pre_solve.d2_joints,
            "move_ccb_along_attached_zz_positive_y_to_pre_solved_D2_joint_goal")) {
        return false;
      }
    } else {
      RCLCPP_INFO(
        this->get_logger(),
        "Planning +Y push along frozen dhz local +Y by %.4f m; axis_in_%s=(%.4f, %.4f, %.4f).",
        zz_y_push_distance_m_,
        planning_frame_.c_str(),
        dhz_positive_y_world.x(),
        dhz_positive_y_world.y(),
        dhz_positive_y_world.z());
      const auto push_pose = offsetCcbAlongDhzLocalAxis(
        ccb_after_down_pose,
        Eigen::Vector3d::UnitY(),
        zz_y_push_distance_m_);
      if (!executeArmPosePlanWithLocalYawSweep(
            push_pose,
            ccb_frame_,
            true,
            "move_ccb_along_attached_zz_positive_y_0p100",
            straight_cartesian_max_step_m_,
            yawSampleSpan(),
            yaw_sample_count_,
            nullptr,
            true,
            min_step_count_,
            refinement_max_steps_)) {
        return false;
      }
    }
    publishReferenceMarkers("after_move_ccb_along_attached_zz_positive_y");

    if (!setDhzAllowance(
          arc7_dhz_allowed_penetration_m_,
          "before_arc7_allow_dhz_0p006")) {
      return false;
    }
    if (!executeCcbArcAroundDhz(
          arc7_center_xyz_dhz_,
          Eigen::Vector3d::UnitX(),
          arc7_roll_angle_rad_,
          "arc7_ccb_rotate_negative_dhz_roll_1p57",
          arc7_cartesian_max_step_m_,
          min_step_count_,
          yaw_sample_count_,
          yawSampleSpan(),
          true,
          arc7_refinement_max_steps_)) {
      return false;
    }
    if (!setDhzAllowance(
          frozen_d_test_scene_allowed_penetration_m_,
          "after_arc7_restore_dhz_0p002")) {
      return false;
    }

    const double arc8_sign = (r_or_l_ == 0) ? -1.0 : 1.0;
    if (!executeCcbArcAroundDhz(
          arc8_center_xyz_dhz_,
          Eigen::Vector3d::UnitZ(),
          arc8_sign * arc8_yaw_angle_abs_rad_,
          "arc8_ccb_rotate_dhz_yaw_by_r_or_l_1p57",
          arc8_cartesian_max_step_m_,
          arc8_min_step_count_,
          1,
          0.0,
          false,
          arc8_refinement_max_steps_)) {
      return false;
    }

    if (!releaseObjectsAndReturnHome()) {
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "dhz_4 sequence completed successfully: gripper opened, E3 detached, and right arm returned to all-zero home.");
    return true;
  }

  void runSequence() {
    if (!waitForService<lite_motion_msgs::srv::PlanArmMotion>(plan_client_, "plan_arm_motion") ||
        !waitForService<lite_motion_msgs::srv::AddSceneObject>(add_object_client_, "add_scene_object") ||
        !waitForService<lite_motion_msgs::srv::RemoveSceneObject>(remove_object_client_, "remove_scene_object") ||
        !waitForService<lite_motion_msgs::srv::AttachSceneObject>(attach_object_client_, "attach_scene_object") ||
        !waitForService<lite_motion_msgs::srv::SetSceneCollisionAllowance>(set_scene_collision_allowance_client_, "set_scene_collision_allowance")) {
      return;
    }

    do {
      bestEffortCleanup();
      if (!runSequenceOnce()) {
        return;
      }
      if (sequence_loop_ && !stop_requested_.load()) {
        sleepAfterAction(loop_pause_sec_);
      }
    } while (sequence_loop_ && !stop_requested_.load());
  }

  void bestEffortCleanup() {
    (void)clearJ7GlobalCollisionIgnore("best_effort_cleanup_clear_j7_global_collision_ignore");
    (void)setGripperAndE3CollisionAllowance(0.0, "best_effort_cleanup_restore_gripper_and_e3");
    (void)setZzGlobalIgnoreAll(0.0, "best_effort_cleanup_clear_zz_ignore");
    (void)setDhzAllowance(0.0, "best_effort_cleanup_clear_dhz_allowance");
    (void)setFrozenDTestSceneAllowance(0.0, "best_effort_cleanup_clear_zz_dhz_scene_allowance");
    (void)setSelfCollisionAllowance(0.0, "best_effort_cleanup_clear_self_collision_allowance");
    (void)setBaseGroupCollisionAllowance(0.0, "best_effort_cleanup_clear_base_group_allowance");
    removeObject(e3_object_id_, false);
    removeObject(zz_object_id_, false);
    removeObject(dhz_object_id_, false);
  }

  void sleepAfterAction(double seconds) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (!stop_requested_.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  void publishReferenceMarkers(const std::string& stage_name = "") {
    if (!ref_marker_pub_) {
      return;
    }
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(clear);
    int id = 0;
    const auto live_d_test_pose = computeDTestPoseInPlanningFrame();
    const auto d_test_pose = makeDTestPoseInPlanningFrame();
    appendFrameMarkers(arr, currentFramePoseStamped(d_test_frame_), d_test_frame_, id);
    appendFrameMarkers(arr, currentFramePoseStamped(tcp_frame_), tcp_frame_, id);
    appendFrameMarkers(arr, currentFramePoseStamped(ccb_frame_), ccb_frame_, id);
    appendFrameMarkers(arr, d_test_pose, d_test_world_pose_valid_ ? "d_test_frozen" : "d_test_live", id);
    appendFrameMarkers(arr, makeDTestPrePose(d_test_pose), "d_test_pre_pos_z", id);
    if (d_test_world_pose_valid_) {
      appendFrameMarkers(arr, live_d_test_pose, "d_test_live", id);
    }
    appendPointMarker(arr, dhzLocalPointToWorld(arc7_center_xyz_dhz_), "arc7_center", id);
    appendPointMarker(arr, dhzLocalPointToWorld(arc8_center_xyz_dhz_), "arc8_center", id);
    ref_marker_pub_->publish(arr);
    if (!stage_name.empty()) {
      logDTestFrameState(stage_name);
    }
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
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::TimerBase::SharedPtr start_timer_;

  std::thread worker_;
  std::atomic_bool stop_requested_{false};

  std::string planning_frame_;
  std::string arm_side_{"right"};
  std::string left_arm_group_name_;
  std::string left_hand_group_name_;
  std::string left_tcp_frame_;
  std::string left_ccb_frame_;
  std::string right_arm_group_name_;
  std::string right_hand_group_name_;
  std::string right_tcp_frame_;
  std::string right_ccb_frame_;
  std::string left_camera_frame_;
  std::string right_camera_frame_;
  std::string right_arm_group_;
  std::string right_hand_group_;
  std::string tcp_frame_;
  std::string ccb_frame_;
  std::string d_test_frame_;
  std::string selected_j7_collision_link_;
  std::string robot_self_collision_ignore_object_id_{"__robot_self_collision_ignore__"};
  std::string robot_self_collision_allowance_object_id_{"__robot_self_collision_allowance__"};
  std::string base_group_collision_allowance_object_id_{"__base_group_collision_allowance__"};
  std::string e3_object_id_;
  std::string e3_mesh_resource_;
  std::string zz_object_id_;
  std::string zz_mesh_resource_;
  std::string dhz_object_id_;
  std::string dhz_mesh_resource_;
  std::vector<double> mesh_scale_;
  std::array<double, 3> d_test_xyz_{};
  std::array<double, 3> d_test_rpy_{};
  std::string zc_target_name_;
  std::string fixed_tail_trajectory_file_;
  std::string fixed_tail_trajectory_dir_;
  double home_planning_timeout_{2.0};
  double home_path_check_step_{0.10};
  int home_ompl_max_plan_attempts_{3};
  std::vector<std::string> base_group_collision_links_{
    "base_link", "cun_1_link", "cun_2_link", "cun_3_link", "cun_4_link"};
  double base_group_collision_allowed_penetration_m_{0.010};
  double gripper_e3_ignore_allowed_penetration_m_{-1.0};
  double return_home_self_collision_allowed_penetration_m_{0.001};
  double return_home_e3_allowed_penetration_m_{0.002};
  double frozen_d_test_scene_allowed_penetration_m_{0.002};
  double arc7_dhz_allowed_penetration_m_{0.006};
  double d_test_pre_offset_along_zz_positive_z_m_{0.012};
  double ccb_approach_distance_m_{0.113};
  double zz_y_push_distance_m_{0.100};
  std::array<double, 3> zz_attach_point_xyz_in_zz_{};
  std::array<double, 3> zz_attach_point_xyz_in_e3_{};
  std::array<double, 3> zz_attach_rpy_in_e3_{};
  std::array<double, 3> arc7_center_xyz_dhz_{};
  double arc7_roll_angle_rad_{-1.5707963267948966};
  bool arc7_pre_solve_d2_enabled_{true};
  std::array<double, 3> arc7_d2_start_xyz_dhz_{0.0, 0.1, -0.1};
  std::array<double, 3> arc7_d2_start_rpy_dhz_{0.0, 0.0, 0.0};
  std::array<double, 2> arc7_d2_yaw_sample_range_{-0.79, 2.3};
  int arc7_d2_yaw_sample_count_{8};
  double arc7_d2_cartesian_max_step_m_{0.08};
  int arc7_d2_min_step_count_{2};
  int arc7_d2_refinement_max_steps_{4};
  int arc7_d2_ik_max_iterations_{180};
  double arc7_d2_ik_pos_tolerance_{0.010};
  double arc7_d2_ik_rot_tolerance_{0.180};
  double arc7_d2_ik_damping_{1.5e-3};
  double arc7_d2_ik_alpha_{0.90};
  double arc7_d2_ik_max_step_norm_{2.50};
  int arc7_d2_ik_max_seed_count_{48};
  int arc7_d2_max_ik_goal_candidates_{12};
  double arc7_d2_planning_timeout_{0.80};
  double arc7_d2_path_check_step_{0.030};
  double arc7_d2_waypoint_pos_tolerance_{0.006};
  double arc7_d2_waypoint_rot_tolerance_{0.120};
  std::array<double, 3> arc8_center_xyz_dhz_{};
  double arc8_yaw_angle_abs_rad_{1.5707963267948966};
  int r_or_l_{0};
  double straight_cartesian_max_step_m_{0.025};
  double arc7_cartesian_max_step_m_{0.042};
  int min_step_count_{4};
  int refinement_max_steps_{8};
  int arc7_refinement_max_steps_{8};
  double arc8_cartesian_max_step_m_{0.004};
  int arc8_min_step_count_{10};
  int arc8_refinement_max_steps_{14};
  std::array<double, 2> yaw_sample_range_{0.79, 2.3};
  int yaw_sample_count_{6};
  bool cartesian_use_request_solver_settings_{true};
  bool cartesian_use_sdls_{true};
  double cartesian_nullspace_gain_{0.20};
  double cartesian_singular_threshold_{0.04};
  double cartesian_branch_jump_max_rad_{0.75};
  double cartesian_branch_jump_norm_max_{1.20};
  double cartesian_waypoint_pos_tolerance_{0.0015};
  double cartesian_waypoint_rot_tolerance_{0.020};
  int cartesian_ik_max_iterations_{1800};
  double cartesian_ik_pos_tolerance_{0.002};
  double cartesian_ik_rot_tolerance_{0.08};
  double cartesian_ik_damping_{1.5e-3};
  double cartesian_ik_alpha_{0.55};
  double cartesian_ik_max_step_norm_{0.20};
  int cartesian_ik_max_seed_count_{60};
  double cartesian_planning_timeout_{1.5};
  double cartesian_path_check_step_{0.020};
  bool detach_zz_during_attached_chain_yaw_sweep_{true};
  bool detach_zz_during_attached_chain_motion_{false};
  geometry_msgs::msg::Pose e3_relative_pose_in_tcp_;
  geometry_msgs::msg::Pose zz_relative_pose_in_tcp_;
  geometry_msgs::msg::PoseStamped zz_initial_world_pose_;
  geometry_msgs::msg::PoseStamped zz_detached_world_pose_before_yaw_;
  bool zz_attached_to_tcp_{false};
  bool zz_detached_world_pose_valid_{false};
  bool zz_initial_world_pose_valid_{false};
  double post_action_wait_sec_{0.05};
  double service_wait_sec_{30.0};
  double response_wait_sec_{30.0};
  bool enable_visualization_{true};
  bool visualization_playback_enabled_{true};
  double visualization_playback_speed_scale_{1.0};
  bool sequence_loop_{false};
  double loop_pause_sec_{0.5};
  geometry_msgs::msg::PoseStamped d_test_world_pose_;
  bool d_test_world_pose_valid_{false};
  std::vector<double> arm_home_positions_;
  std::vector<double> hand_open_positions_;
  std::vector<double> hand_close_positions_;
  std::vector<double> right_observe_joint_positions_;
  std::vector<double> left_observe_joint_positions_;
  std::vector<double> observe_arm_positions_;
  std::vector<double> observe_hand_positions_;
  std::vector<double> right_arm_state_;
  std::vector<double> right_hand_state_;
};

}  // namespace lite_motion_planner

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<lite_motion_planner::Dhz4Node>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
