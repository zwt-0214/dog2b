#include "lite_motion_planner/robot_model.hpp"

#include <lite_motion_msgs/srv/add_scene_object.hpp>
#include <lite_motion_msgs/srv/attach_scene_object.hpp>
#include <lite_motion_msgs/srv/plan_arm_motion.hpp>
#include <lite_motion_msgs/srv/remove_scene_object.hpp>
#include <lite_motion_msgs/srv/set_scene_collision_allowance.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
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
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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

class HaomuCatchNode : public rclcpp::Node {
public:
  HaomuCatchNode()
  : Node("haomu_catch") {
    const auto urdf_path = this->declare_parameter<std::string>("urdf_path", "");
    const auto srdf_path = this->declare_parameter<std::string>("srdf_path", "");
    const auto mesh_package_dirs = this->declare_parameter<std::vector<std::string>>(
      "mesh_package_dirs", std::vector<std::string>());

    planning_frame_ = this->declare_parameter<std::string>("planning_frame", "base_link");
    arm_side_ = normalizeArmSide(this->declare_parameter<std::string>("haomu.arm_side", "right"));
    left_arm_group_name_ = this->declare_parameter<std::string>("haomu.left_arm_group", "left_arm");
    left_hand_group_name_ = this->declare_parameter<std::string>("haomu.left_hand_group", "left_hand");
    left_tcp_frame_ = this->declare_parameter<std::string>("haomu.left_tcp_frame", "left_tcp");
    right_arm_group_name_ = this->declare_parameter<std::string>("haomu.right_arm_group", "right_arm");
    right_hand_group_name_ = this->declare_parameter<std::string>("haomu.right_hand_group", "right_hand");
    right_tcp_frame_ = this->declare_parameter<std::string>("haomu.right_tcp_frame", "right_tcp");
    selectArm(arm_side_);
    ee_frame_ = this->declare_parameter<std::string>("haomu.ee_frame", ee_frame_);
    object_id_ = this->declare_parameter<std::string>("haomu.object_id", "E3");
    object_mesh_resource_ = this->declare_parameter<std::string>("haomu.object_mesh_resource", "/home/zwt/E3.STL");
    mesh_scale_ = this->declare_parameter<std::vector<double>>(
      "haomu.object_mesh_scale", std::vector<double>{0.001, 0.001, 0.001});
    p_test_xyz_ = declareVec3Param("haomu.p_test_xyz", {0.4, 0.2, 0.1});
    p_test_rpy_ = declareVec3Param("haomu.p_test_rpy", {0.0, 0.0, 0.0});
    pregrasp_distance_m_ = std::max(1e-4, this->declare_parameter<double>("haomu.pregrasp_distance_m", 0.04));
    pregrasp_reference_joint_ = this->declare_parameter<std::string>(
      "haomu.pregrasp_reference_joint", pregrasp_reference_joint_);
    grasp_target_ignore_links_ = this->declare_parameter<std::vector<std::string>>(
      "haomu.grasp_target_ignore_links",
      grasp_target_ignore_links_);
    alignSelectedArmParameters();
    grasp_target_restore_allowed_penetration_m_ = std::max(
      0.0,
      this->declare_parameter<double>("haomu.grasp_target_restore_allowed_penetration", 0.005));
    zc_f_xyz_ = declareVec3Param("haomu.zc_f_xyz", {0.185, 0.0, 0.0725});
    zc_f_rpy_ = declareVec3Param("haomu.zc_f_rpy", {0.0, 0.0, 0.0});
    zc_m_xyz_ = declareVec3Param("haomu.zc_m_xyz", {0.085, 0.0, 0.0725});
    zc_m_rpy_ = declareVec3Param("haomu.zc_m_rpy", {0.0, 0.0, 0.0});
    zc_l_xyz_ = declareVec3Param("haomu.zc_l_xyz", {-0.0016, 0.050, 0.0725});
    zc_l_rpy_ = declareVec3Param("haomu.zc_l_rpy", {0.0, 0.0, 0.0});
    zc_r_xyz_ = declareVec3Param("haomu.zc_r_xyz", {-0.0016, -0.050, 0.0725});
    zc_r_rpy_ = declareVec3Param("haomu.zc_r_rpy", {0.0, 0.0, 0.0});
    final_target_name_ = this->declare_parameter<std::string>("haomu.final_target", "zc_f");
    lift_distance_m_ = this->declare_parameter<double>("haomu.lift_distance_m", 0.11);
    cartesian_max_step_m_ = this->declare_parameter<double>("haomu.cartesian_max_step_m", 0.08);
    cartesian_use_sdls_ = this->declare_parameter<bool>("haomu.cartesian_use_sdls", true);
    cartesian_nullspace_gain_ = std::max(
      0.0,
      this->declare_parameter<double>("haomu.cartesian_nullspace_gain", 0.20));
    cartesian_singular_threshold_ = std::max(
      1e-6,
      this->declare_parameter<double>("haomu.cartesian_singular_threshold", 0.04));
    cartesian_branch_jump_max_rad_ = std::max(
      1e-6,
      this->declare_parameter<double>("haomu.cartesian_branch_jump_max_rad", 0.75));
    cartesian_branch_jump_norm_max_ = std::max(
      1e-6,
      this->declare_parameter<double>("haomu.cartesian_branch_jump_norm_max", 1.20));
    cartesian_waypoint_pos_tolerance_ = std::max(
      1e-6,
      this->declare_parameter<double>("haomu.cartesian_waypoint_pos_tolerance", 0.0015));
    cartesian_waypoint_rot_tolerance_ = std::max(
      1e-6,
      this->declare_parameter<double>("haomu.cartesian_waypoint_rot_tolerance", 0.020));
    grasp_yaw_tolerance_rad_ = std::max(0.0, this->declare_parameter<double>("haomu.grasp_yaw_tolerance_rad", 2.0 * M_PI));
    grasp_yaw_sample_count_ = std::max<int>(1, this->declare_parameter<int>("haomu.grasp_yaw_sample_count", 8));
    final_base_yaw_tolerance_rad_ = std::max(0.0, this->declare_parameter<double>("haomu.final_base_yaw_tolerance_rad", 2.0 * M_PI));
    final_base_yaw_sample_count_ = std::max<int>(1, this->declare_parameter<int>("haomu.final_base_yaw_sample_count", 32));
    final_approach_height_m_ = std::max(
      1e-4,
      this->declare_parameter<double>("haomu.final_approach_height_m", 0.12));
    final_drop_distance_m_ = std::max(
      1e-4,
      this->declare_parameter<double>("haomu.final_drop_distance_m", 0.12));
    final_drop_cartesian_max_step_m_ = std::max(
      1e-4,
      this->declare_parameter<double>("haomu.final_drop_cartesian_max_step_m", 0.02));
    home_planning_timeout_ = std::max(0.05, this->declare_parameter<double>("haomu.home_planning_timeout", 2.0));
    home_path_check_step_ = std::max(0.005, this->declare_parameter<double>("haomu.home_path_check_step", 0.10));
    home_ompl_max_plan_attempts_ = std::max<int>(1, this->declare_parameter<int>("haomu.home_ompl_max_plan_attempts", 3));
    remove_object_before_return_home_ = this->declare_parameter<bool>("haomu.remove_object_before_return_home", true);
    final_relaxed_object_ids_ = this->declare_parameter<std::vector<std::string>>(
      "haomu.final_relaxed_scene_collision_object_ids", std::vector<std::string>{object_id_});
    final_relaxed_links_ = this->declare_parameter<std::vector<std::string>>(
      "haomu.final_relaxed_scene_collision_links",
      std::vector<std::string>{"cun_1_link", "cun_2_link", "cun_3_link", "cun_4_link"});
    return_home_relaxed_object_ids_ = this->declare_parameter<std::vector<std::string>>(
      "haomu.return_home_relaxed_scene_collision_object_ids",
      std::vector<std::string>{object_id_});
    return_home_relaxed_links_ = this->declare_parameter<std::vector<std::string>>(
      "haomu.return_home_relaxed_scene_collision_links",
      std::vector<std::string>{"cun_1_link", "cun_2_link", "cun_3_link", "cun_4_link"});
    return_home_relaxed_allowed_penetration_m_ = std::max(0.0, this->declare_parameter<double>(
      "haomu.return_home_relaxed_scene_collision_allowed_penetration", 0.005));
    return_home_gripper_e3_object_ids_ = this->declare_parameter<std::vector<std::string>>(
      "haomu.return_home_gripper_e3_collision_object_ids",
      std::vector<std::string>{object_id_});
    return_home_gripper_e3_links_ = this->declare_parameter<std::vector<std::string>>(
      "haomu.return_home_gripper_e3_collision_links",
      std::vector<std::string>{
        "zhua_left_1_link", "zhua_left_2_link", "zhua_right_1_link", "zhua_right_2_link"});
    return_home_gripper_e3_allowed_penetration_m_ = std::max(0.0, this->declare_parameter<double>(
      "haomu.return_home_gripper_e3_collision_allowed_penetration", 0.002));
    return_home_j7_retreat_frame_ = this->declare_parameter<std::string>(
      "haomu.return_home_j7_retreat_frame", return_home_j7_retreat_frame_);
    if (return_home_j7_retreat_frame_ == defaultReturnHomeJ7RetreatFrame(
          arm_side_ == "left" ? "right" : "left")) {
      return_home_j7_retreat_frame_ = defaultReturnHomeJ7RetreatFrame(arm_side_);
    }
    return_home_j7_retreat_distance_m_ = std::max(
      0.0,
      this->declare_parameter<double>("haomu.return_home_j7_retreat_distance_m", 0.04));
    return_home_j7_retreat_cartesian_max_step_m_ = std::max(
      1e-4,
      this->declare_parameter<double>("haomu.return_home_j7_retreat_cartesian_max_step_m", 0.01));
    remove_object_after_release_ = this->declare_parameter<bool>("haomu.remove_object_after_release", false);
    post_action_wait_sec_ = this->declare_parameter<double>("haomu.post_action_wait_sec", 0.05);
    hand_open_positions_ = this->declare_parameter<std::vector<double>>(
      "haomu.hand_open_positions", std::vector<double>{0.0, 0.0});
    hand_close_positions_ = this->declare_parameter<std::vector<double>>(
      "haomu.hand_close_positions", std::vector<double>{-0.019, 0.019});
    service_wait_sec_ = this->declare_parameter<double>("haomu.service_wait_sec", 30.0);
    response_wait_sec_ = this->declare_parameter<double>("haomu.response_wait_sec", 30.0);
    const bool enable_visualization = this->declare_parameter<bool>("enable_visualization", true);
    visualization_playback_enabled_ = enable_visualization && this->declare_parameter<bool>("enable_joint_state_playback", true);
    visualization_playback_speed_scale_ = std::max(1e-3, this->declare_parameter<double>("joint_state_playback_speed_scale", 1.0));
    sequence_loop_ = this->declare_parameter<bool>("haomu.sequence_loop", true);
    loop_pause_sec_ = std::max(0.0, this->declare_parameter<double>("haomu.loop_pause_sec", 0.5));

    robot_model_ = std::make_unique<RobotModel>(urdf_path, srdf_path, mesh_package_dirs);
    const auto& arm_group = robot_model_->getPlanningGroup(right_arm_group_);
    const auto& hand_group = robot_model_->getPlanningGroup(right_hand_group_);
    arm_home_positions_ = this->declare_parameter<std::vector<double>>(
      "haomu.arm_home_positions", std::vector<double>(arm_group.joint_names.size(), 0.0));

    right_arm_state_ = arm_home_positions_;
    right_hand_state_ = hand_open_positions_;

    if (arm_home_positions_.size() != arm_group.joint_names.size()) {
      throw std::runtime_error("haomu.arm_home_positions size does not match selected arm group dof");
    }
    if (hand_open_positions_.size() != hand_group.joint_names.size()) {
      throw std::runtime_error("haomu.hand_open_positions size does not match selected hand group dof");
    }
    if (hand_close_positions_.size() != right_hand_state_.size()) {
      throw std::runtime_error("haomu.hand_close_positions size does not match selected hand group dof");
    }
    if (!mesh_scale_.empty() && mesh_scale_.size() != 3) {
      throw std::runtime_error("haomu.object_mesh_scale must contain exactly 3 values");
    }
    RCLCPP_INFO(
      this->get_logger(),
      "Selected haomu catch arm: arm_side=%s arm_group=%s hand_group=%s ee_frame=%s",
      arm_side_.c_str(),
      right_arm_group_.c_str(),
      right_hand_group_.c_str(),
      ee_frame_.c_str());

    plan_client_ = this->create_client<lite_motion_msgs::srv::PlanArmMotion>("plan_arm_motion");
    add_object_client_ = this->create_client<lite_motion_msgs::srv::AddSceneObject>("add_scene_object");
    attach_object_client_ = this->create_client<lite_motion_msgs::srv::AttachSceneObject>("attach_scene_object");
    remove_object_client_ = this->create_client<lite_motion_msgs::srv::RemoveSceneObject>("remove_scene_object");
    set_scene_collision_allowance_client_ = this->create_client<lite_motion_msgs::srv::SetSceneCollisionAllowance>(
      "set_scene_collision_allowance");

    rclcpp::QoS latched_qos(1);
    latched_qos.transient_local();
    latched_qos.reliable();
    ref_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "haomu_catch/reference_frames", latched_qos);

    start_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      [this]() {
        start_timer_->cancel();
        publishReferenceMarkers();
        worker_ = std::thread([this]() { runSequence(); });
      });
  }

  ~HaomuCatchNode() override {
    stop_requested_.store(true);
    if (worker_.joinable()) {
      worker_.join();
    }
  }

private:
  std::string normalizeArmSide(const std::string& side) const {
    if (side == "0" || side == "left" || side == "l") {
      return "left";
    }
    if (side == "1" || side == "right" || side == "r") {
      return "right";
    }
    throw std::runtime_error("haomu.arm_side must be left/right or 0/1");
  }

  void selectArm(const std::string& side) {
    if (side == "left") {
      right_arm_group_ = left_arm_group_name_;
      right_hand_group_ = left_hand_group_name_;
      ee_frame_ = left_tcp_frame_;
      return_home_j7_retreat_frame_ = defaultReturnHomeJ7RetreatFrame(side);
      pregrasp_reference_joint_ = defaultPregraspReferenceJoint(side);
      grasp_target_ignore_links_ = defaultGraspTargetIgnoreLinks(side);
      return;
    }

    right_arm_group_ = right_arm_group_name_;
    right_hand_group_ = right_hand_group_name_;
    ee_frame_ = right_tcp_frame_;
    return_home_j7_retreat_frame_ = defaultReturnHomeJ7RetreatFrame(side);
    pregrasp_reference_joint_ = defaultPregraspReferenceJoint(side);
    grasp_target_ignore_links_ = defaultGraspTargetIgnoreLinks(side);
  }

  std::string defaultReturnHomeJ7RetreatFrame(const std::string& side) const {
    return side == "left" ? "arm_left_J7_link" : "arm_right_J7_link";
  }

  std::string defaultPregraspReferenceJoint(const std::string& side) const {
    return side == "left" ? "arm_left_J1_joint" : "arm_right_J1_joint";
  }

  std::vector<std::string> defaultGraspTargetIgnoreLinks(const std::string& side) const {
    if (side == "left") {
      return {"left_tcp", "left_ccb", "arm_left_J7_link", "zhua_left_1_link", "zhua_left_2_link"};
    }
    return {"right_tcp", "right_ccb", "arm_right_J7_link", "zhua_right_1_link", "zhua_right_2_link"};
  }

  void alignSelectedArmParameters() {
    const std::string other_side = arm_side_ == "left" ? "right" : "left";
    const std::string selected_tcp = arm_side_ == "left" ? left_tcp_frame_ : right_tcp_frame_;
    const std::string other_tcp = arm_side_ == "left" ? right_tcp_frame_ : left_tcp_frame_;
    if (ee_frame_ == other_tcp) {
      ee_frame_ = selected_tcp;
    }
    if (pregrasp_reference_joint_ == defaultPregraspReferenceJoint(other_side)) {
      pregrasp_reference_joint_ = defaultPregraspReferenceJoint(arm_side_);
    }
    if (grasp_target_ignore_links_ == defaultGraspTargetIgnoreLinks(other_side)) {
      grasp_target_ignore_links_ = defaultGraspTargetIgnoreLinks(arm_side_);
    }
  }

  std::array<double, 3> declareVec3Param(
    const std::string& name,
    const std::array<double, 3>& default_value) {
    auto values = this->declare_parameter<std::vector<double>>(
      name,
      std::vector<double>{default_value[0], default_value[1], default_value[2]});
    if (values.size() != 3) {
      throw std::runtime_error(name + " must contain exactly 3 values");
    }
    return {values[0], values[1], values[2]};
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

  geometry_msgs::msg::PoseStamped offsetAlongLocalZ(
    const geometry_msgs::msg::PoseStamped& in,
    double distance_m) const {
    geometry_msgs::msg::PoseStamped out = in;
    const Eigen::Quaterniond q = quaternionEigenFromMsg(in.pose.orientation);
    const Eigen::Vector3d dz = q.toRotationMatrix() * Eigen::Vector3d::UnitZ() * distance_m;
    out.pose.position.x += dz.x();
    out.pose.position.y += dz.y();
    out.pose.position.z += dz.z();
    return out;
  }

  geometry_msgs::msg::PoseStamped offsetAlongBaseZ(
    const geometry_msgs::msg::PoseStamped& in,
    double distance_m) const {
    geometry_msgs::msg::PoseStamped out = in;
    out.pose.position.z += distance_m;
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

  geometry_msgs::msg::PoseStamped rotatePoseAboutBaseZ(
    const geometry_msgs::msg::PoseStamped& in,
    double yaw_offset_rad) const {
    geometry_msgs::msg::PoseStamped out = in;
    const Eigen::Quaterniond base_q = quaternionEigenFromMsg(in.pose.orientation);
    const Eigen::Quaterniond yaw_q(Eigen::AngleAxisd(yaw_offset_rad, Eigen::Vector3d::UnitZ()));
    const Eigen::Quaterniond q_new = (yaw_q * base_q).normalized();
    out.pose.orientation.x = q_new.x();
    out.pose.orientation.y = q_new.y();
    out.pose.orientation.z = q_new.z();
    out.pose.orientation.w = q_new.w();
    return out;
  }

  std::vector<double> generateYawOffsets(double tolerance_rad, int sample_count) const {
    std::vector<double> offsets;
    offsets.push_back(0.0);
    if (sample_count <= 1 || tolerance_rad < 1e-6) {
      return offsets;
    }

    const double tol = std::min(tolerance_rad, 2.0 * M_PI);
    const bool full_circle = tol >= (2.0 * M_PI - 1e-3);
    if (full_circle) {
      const double step = 2.0 * M_PI / static_cast<double>(sample_count);
      for (int k = 1; static_cast<int>(offsets.size()) < sample_count; ++k) {
        offsets.push_back(wrapToPi(static_cast<double>(k) * step));
        if (static_cast<int>(offsets.size()) >= sample_count) {
          break;
        }
        offsets.push_back(wrapToPi(-static_cast<double>(k) * step));
      }
      if (static_cast<int>(offsets.size()) > sample_count) {
        offsets.resize(static_cast<size_t>(sample_count));
      }
      return offsets;
    }

    const double half_span = 0.5 * tol;
    const int side_count = std::max(1, sample_count / 2);
    const double step = half_span / static_cast<double>(side_count);
    for (int k = 1; static_cast<int>(offsets.size()) < sample_count; ++k) {
      const double delta = std::min(half_span, static_cast<double>(k) * step);
      offsets.push_back(delta);
      if (static_cast<int>(offsets.size()) >= sample_count) {
        break;
      }
      offsets.push_back(-delta);
    }
    return offsets;
  }

  std::vector<double> generateGraspYawOffsets() const {
    return generateYawOffsets(grasp_yaw_tolerance_rad_, grasp_yaw_sample_count_);
  }

  std::vector<double> generateFinalBaseYawOffsets() const {
    return generateYawOffsets(final_base_yaw_tolerance_rad_, final_base_yaw_sample_count_);
  }

  pinocchio::SE3 currentFramePose(const std::string& frame_name) {
    const Eigen::VectorXd q_group = Eigen::Map<const Eigen::VectorXd>(
      right_arm_state_.data(), static_cast<Eigen::Index>(right_arm_state_.size()));
    const Eigen::VectorXd q_full = robot_model_->groupToFull(
      right_arm_group_, q_group, robot_model_->neutralConfiguration());
    pinocchio::forwardKinematics(robot_model_->model(), robot_model_->data(), q_full);
    pinocchio::updateFramePlacements(robot_model_->model(), robot_model_->data());
    return robot_model_->data().oMf[robot_model_->getFrameIdChecked(frame_name)];
  }

  pinocchio::SE3 currentEePose() {
    return currentFramePose(ee_frame_);
  }

  geometry_msgs::msg::Pose computeAttachRelativePose(
    const geometry_msgs::msg::PoseStamped& object_world_pose) {
    const pinocchio::SE3 ee_world = currentEePose();
    const pinocchio::SE3 object_world = poseMsgToSE3(object_world_pose.pose);
    return poseMsgFromSE3(ee_world.actInv(object_world));
  }

  bool executeArmPosePlanWithGraspYawSweep(
    const geometry_msgs::msg::PoseStamped& target_pose,
    bool use_cartesian_path,
    const std::string& stage_name,
    geometry_msgs::msg::PoseStamped* executed_pose = nullptr) {
    const auto yaw_offsets = generateGraspYawOffsets();
    for (size_t i = 0; i < yaw_offsets.size(); ++i) {
      const auto candidate_pose = rotatePoseAboutLocalZ(target_pose, yaw_offsets[i]);
      if (yaw_offsets.size() > 1) {
        RCLCPP_INFO(
          this->get_logger(),
          "Stage [%s] trying local-yaw sample %zu/%zu (offset=%.4f rad)",
          stage_name.c_str(),
          i + 1,
          yaw_offsets.size(),
          yaw_offsets[i]);
      }
      if (executeArmPosePlan(candidate_pose, use_cartesian_path, stage_name)) {
        if (executed_pose != nullptr) {
          *executed_pose = candidate_pose;
        }
        return true;
      }
    }
    return false;
  }

  geometry_msgs::msg::PoseStamped computePregraspPose() {
    const auto p_test = makePoseStamped(p_test_xyz_, p_test_rpy_);

    const Eigen::VectorXd q0 = robot_model_->neutralConfiguration();
    pinocchio::forwardKinematics(robot_model_->model(), robot_model_->data(), q0);
    pinocchio::updateFramePlacements(robot_model_->model(), robot_model_->data());

    const auto jid = robot_model_->model().getJointId(pregrasp_reference_joint_);
    if (jid == 0 || jid >= static_cast<decltype(jid)>(robot_model_->model().njoints)) {
      throw std::runtime_error("Unknown pregrasp reference joint: " + pregrasp_reference_joint_);
    }

    const Eigen::Vector3d j1_world = robot_model_->data().oMi[jid].translation();
    Eigen::Vector2d dir_xz(
      p_test.pose.position.x - j1_world.x(),
      p_test.pose.position.z - j1_world.z());

    if (dir_xz.norm() < 1e-9) {
      const Eigen::Quaterniond q = quaternionEigenFromMsg(p_test.pose.orientation);
      const Eigen::Vector3d local_x = q.toRotationMatrix() * Eigen::Vector3d::UnitX();
      dir_xz = Eigen::Vector2d(local_x.x(), local_x.z());
    }

    if (dir_xz.norm() < 1e-9) {
      dir_xz = Eigen::Vector2d(1.0, 0.0);
    }
    dir_xz.normalize();

    geometry_msgs::msg::PoseStamped out = p_test;
    out.pose.position.x -= pregrasp_distance_m_ * dir_xz.x();
    out.pose.position.z -= pregrasp_distance_m_ * dir_xz.y();
    out.pose.position.y = p_test.pose.position.y;
    return out;
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
    appendFrameMarkers(arr, makePoseStamped(p_test_xyz_, p_test_rpy_), "p_test", id);
    appendFrameMarkers(arr, computePregraspPose(), "pregrasp", id);
    appendFrameMarkers(arr, makePoseStamped(zc_f_xyz_, zc_f_rpy_), "zc_f", id);
    appendFrameMarkers(arr, makePoseStamped(zc_m_xyz_, zc_m_rpy_), "zc_m", id);
    appendFrameMarkers(arr, makePoseStamped(zc_l_xyz_, zc_l_rpy_), "zc_l", id);
    appendFrameMarkers(arr, makePoseStamped(zc_r_xyz_, zc_r_rpy_), "zc_r", id);
    ref_marker_pub_->publish(arr);
  }

  geometry_msgs::msg::PoseStamped selectedFinalTargetPose(std::string& target_name_out) const {
    if (final_target_name_ == "zc_m") {
      target_name_out = "zc_m";
      return makePoseStamped(zc_m_xyz_, zc_m_rpy_);
    }
    if (final_target_name_ == "zc_l") {
      target_name_out = "zc_l";
      return makePoseStamped(zc_l_xyz_, zc_l_rpy_);
    }
    if (final_target_name_ == "zc_r") {
      target_name_out = "zc_r";
      return makePoseStamped(zc_r_xyz_, zc_r_rpy_);
    }
    if (final_target_name_ != "zc_f") {
      RCLCPP_WARN(
        this->get_logger(),
        "Unknown haomu.final_target [%s], falling back to zc_f",
        final_target_name_.c_str());
    }
    target_name_out = "zc_f";
    return makePoseStamped(zc_f_xyz_, zc_f_rpy_);
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

  template<typename ServiceT>
  bool waitForService(
    const typename rclcpp::Client<ServiceT>::SharedPtr& client,
    const std::string& service_name) {
    const auto timeout = std::chrono::duration<double>(service_wait_sec_);
    if (client->wait_for_service(timeout)) {
      return true;
    }

    RCLCPP_ERROR(
      this->get_logger(),
      "Service [%s] not available within %.2f s",
      service_name.c_str(),
      service_wait_sec_);
    return false;
  }

  template<typename ServiceT>
  std::shared_ptr<typename ServiceT::Response> callService(
    const typename rclcpp::Client<ServiceT>::SharedPtr& client,
    const std::shared_ptr<typename ServiceT::Request>& request,
    const std::string& service_name) {
    auto future = client->async_send_request(request);
    const auto timeout = std::chrono::duration<double>(response_wait_sec_);
    if (future.wait_for(timeout) != std::future_status::ready) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Timed out waiting for service [%s] response after %.2f s",
        service_name.c_str(),
        response_wait_sec_);
      return nullptr;
    }
    return future.get();
  }

  bool addE3Object(const geometry_msgs::msg::PoseStamped& object_pose) {
    auto req = std::make_shared<lite_motion_msgs::srv::AddSceneObject::Request>();
    req->object_id = object_id_;
    req->mesh_resource = object_mesh_resource_;
    req->pose = object_pose;
    req->collision_pose = object_pose.pose;
    req->mesh_scale = mesh_scale_;

    auto resp = callService<lite_motion_msgs::srv::AddSceneObject>(
      add_object_client_, req, "add_scene_object");
    if (!resp) {
      return false;
    }
    if (!resp->success) {
      RCLCPP_ERROR(this->get_logger(), "Add scene object failed: %s", resp->message.c_str());
      return false;
    }

    object_exists_ = true;
    object_attached_ = false;

    RCLCPP_INFO(
      this->get_logger(),
      "Scene object [%s] loaded at p_test from [%s]",
      object_id_.c_str(),
      object_mesh_resource_.c_str());
    sleepAfterAction(post_action_wait_sec_);
    return true;
  }


  bool removeE3Object() {
    auto req = std::make_shared<lite_motion_msgs::srv::RemoveSceneObject::Request>();
    req->object_id = object_id_;

    auto resp = callService<lite_motion_msgs::srv::RemoveSceneObject>(
      remove_object_client_, req, "remove_scene_object");
    if (!resp) {
      return false;
    }
    if (!resp->success) {
      RCLCPP_ERROR(this->get_logger(), "Remove scene object failed: %s", resp->message.c_str());
      return false;
    }

    object_exists_ = false;
    object_attached_ = false;

    RCLCPP_INFO(this->get_logger(), "Scene object [%s] removed from collision scene", object_id_.c_str());
    sleepAfterAction(post_action_wait_sec_);
    return true;
  }

  bool setObjectAttached(bool attach, const geometry_msgs::msg::Pose* relative_pose = nullptr) {
    auto req = std::make_shared<lite_motion_msgs::srv::AttachSceneObject::Request>();
    req->object_id = object_id_;
    req->attach = attach;
    req->frame_id = ee_frame_;
    if (relative_pose != nullptr) {
      req->relative_pose = *relative_pose;
    } else {
      req->relative_pose.orientation.w = 1.0;
    }

    auto resp = callService<lite_motion_msgs::srv::AttachSceneObject>(
      attach_object_client_, req, "attach_scene_object");
    if (!resp) {
      return false;
    }
    if (!resp->success) {
      RCLCPP_ERROR(
        this->get_logger(),
        "%s object failed: %s",
        attach ? "Attach" : "Detach",
        resp->message.c_str());
      return false;
    }

    object_attached_ = attach;

    RCLCPP_INFO(
      this->get_logger(),
      "%s [%s] %s [%s]",
      attach ? "Attached" : "Detached",
      object_id_.c_str(),
      attach ? "to" : "from",
      ee_frame_.c_str());
    sleepAfterAction(post_action_wait_sec_);
    return true;
  }

  bool setRuntimeSceneCollisionAllowance(
    const std::vector<std::string>& object_ids,
    const std::vector<std::string>& link_names,
    double allowed_penetration_m,
    const std::string& stage_name,
    bool warn_only = false) {
    if (object_ids.empty() || link_names.empty()) {
      RCLCPP_WARN(
        this->get_logger(),
        "Stage [%s] skipped runtime collision allowance because object_ids or link_names is empty.",
        stage_name.c_str());
      return warn_only;
    }

    auto req = std::make_shared<lite_motion_msgs::srv::SetSceneCollisionAllowance::Request>();
    req->object_ids = object_ids;
    req->link_names = link_names;
    req->allowed_penetration = allowed_penetration_m;

    auto resp = callService<lite_motion_msgs::srv::SetSceneCollisionAllowance>(
      set_scene_collision_allowance_client_, req, "set_scene_collision_allowance");
    if (!resp || !resp->success) {
      const char* msg = resp ? resp->message.c_str() : "no response";
      if (warn_only) {
        RCLCPP_WARN(
          this->get_logger(),
          "Set scene collision allowance failed at stage [%s]: %s",
          stage_name.c_str(),
          msg);
        return true;
      }
      RCLCPP_ERROR(
        this->get_logger(),
        "Set scene collision allowance failed at stage [%s]: %s",
        stage_name.c_str(),
        msg);
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

  bool applyFinalTargetCollisionAllowance(const std::string& stage_name) {
    std::vector<std::string> final_links = final_relaxed_links_;
    if (std::find(final_links.begin(), final_links.end(), "base_link") == final_links.end()) {
      final_links.push_back("base_link");
    }
    if (!setRuntimeSceneCollisionAllowance(
          final_relaxed_object_ids_,
          final_links,
          -1.0,
          stage_name)) {
      return false;
    }
    final_relaxed_links_active_ = std::move(final_links);
    final_target_collision_allowance_active_ = true;
    return true;
  }

  void clearFinalTargetCollisionAllowanceBestEffort(const std::string& stage_name) {
    if (!final_target_collision_allowance_active_) {
      return;
    }
    (void)setRuntimeSceneCollisionAllowance(
      final_relaxed_object_ids_,
      final_relaxed_links_active_.empty() ? final_relaxed_links_ : final_relaxed_links_active_,
      0.0,
      stage_name,
      true);
    final_relaxed_links_active_.clear();
    final_target_collision_allowance_active_ = false;
  }

  bool applyReturnHomeCollisionAllowance(const std::string& stage_name) {
    if (!return_home_relaxed_object_ids_.empty() && !return_home_relaxed_links_.empty()) {
      std::vector<std::string> return_home_links = return_home_relaxed_links_;
      if (std::find(return_home_links.begin(), return_home_links.end(), "base_link") == return_home_links.end()) {
        return_home_links.push_back("base_link");
      }
      if (!setRuntimeSceneCollisionAllowance(
            return_home_relaxed_object_ids_,
            return_home_links,
            -1.0,
            stage_name)) {
        return false;
      }
      return_home_relaxed_links_active_ = std::move(return_home_links);
      return_home_collision_allowance_active_ = true;
    }
    if (!return_home_gripper_e3_object_ids_.empty() && !return_home_gripper_e3_links_.empty() &&
        return_home_gripper_e3_allowed_penetration_m_ > 0.0) {
      if (!setRuntimeSceneCollisionAllowance(
            return_home_gripper_e3_object_ids_,
            return_home_gripper_e3_links_,
            return_home_gripper_e3_allowed_penetration_m_,
            stage_name)) {
        return false;
      }
      return_home_gripper_e3_collision_allowance_active_ = true;
    }
    return true;
  }

  void clearReturnHomeCollisionAllowanceBestEffort(const std::string& stage_name) {
    if (return_home_gripper_e3_collision_allowance_active_) {
      (void)setRuntimeSceneCollisionAllowance(
        return_home_gripper_e3_object_ids_,
        return_home_gripper_e3_links_,
        0.0,
        stage_name,
        true);
      return_home_gripper_e3_collision_allowance_active_ = false;
    }
    if (return_home_collision_allowance_active_) {
      (void)setRuntimeSceneCollisionAllowance(
        return_home_relaxed_object_ids_,
        return_home_relaxed_links_active_.empty() ? return_home_relaxed_links_ : return_home_relaxed_links_active_,
        0.0,
        stage_name,
        true);
      return_home_relaxed_links_active_.clear();
      return_home_collision_allowance_active_ = false;
    }
  }

  bool applyGraspTargetIgnore(const std::string& stage_name) {
    if (grasp_target_ignore_links_.empty()) {
      return true;
    }
    if (!setRuntimeSceneCollisionAllowance(
          {object_id_},
          grasp_target_ignore_links_,
          -1.0,
          stage_name)) {
      return false;
    }
    grasp_target_ignore_active_ = true;
    return true;
  }

  void clearGraspTargetIgnoreBestEffort(const std::string& stage_name) {
    if (!grasp_target_ignore_active_) {
      return;
    }
    (void)setRuntimeSceneCollisionAllowance(
      {object_id_},
      grasp_target_ignore_links_,
      grasp_target_restore_allowed_penetration_m_,
      stage_name,
      true);
    grasp_target_ignore_active_ = false;
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
    bool use_cartesian_path,
    const std::string& stage_name,
    double cartesian_max_step_override = 0.0) {
    return executeArmPosePlanForFrame(
      target_pose, ee_frame_, use_cartesian_path, stage_name, cartesian_max_step_override);
  }

  bool executeArmPosePlanForFrame(
    const geometry_msgs::msg::PoseStamped& target_pose,
    const std::string& frame_name,
    bool use_cartesian_path,
    const std::string& stage_name,
    double cartesian_max_step_override = 0.0) {
    auto req = std::make_shared<lite_motion_msgs::srv::PlanArmMotion::Request>();
    req->group_name = right_arm_group_;
    req->ee_frame = frame_name;
    req->use_pose_goal = true;
    req->target_pose = target_pose;
    req->start_joint_positions = right_arm_state_;
    req->use_cartesian_path = use_cartesian_path;
    req->cartesian_max_step = cartesian_max_step_override > 0.0 ?
      cartesian_max_step_override : cartesian_max_step_m_;
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

    auto resp = callService<lite_motion_msgs::srv::PlanArmMotion>(
      plan_client_, req, "plan_arm_motion");
    if (!resp) {
      return false;
    }
    if (!resp->success) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Arm planning failed at stage [%s]: %s",
        stage_name.c_str(),
        resp->message.c_str());
      return false;
    }
    if (resp->trajectory.points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Arm planning returned an empty trajectory at stage [%s]", stage_name.c_str());
      return false;
    }

    right_arm_state_ = resp->trajectory.points.back().positions;
    const double wait_sec = effectiveExecutionWaitSec(resp->trajectory);
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] trajectory accepted, waiting %.3f s for execution/playback",
      stage_name.c_str(),
      wait_sec);
    sleepAfterAction(wait_sec);
    return true;
  }

  bool armStateMatches(const std::vector<double>& target_joints) const {
    if (target_joints.size() != right_arm_state_.size()) {
      return false;
    }
    double max_err = 0.0;
    for (size_t i = 0; i < target_joints.size(); ++i) {
      max_err = std::max(max_err, std::abs(target_joints[i] - right_arm_state_[i]));
    }
    return max_err < 1e-6;
  }

  bool executeReturnHomeJ7Retreat(const std::string& stage_name) {
    if (return_home_j7_retreat_distance_m_ <= 0.0) {
      return true;
    }
    const pinocchio::SE3 j7_world = currentFramePose(return_home_j7_retreat_frame_);
    const pinocchio::SE3 target_world =
      j7_world.act(pinocchio::SE3(
        Eigen::Matrix3d::Identity(),
        Eigen::Vector3d(-return_home_j7_retreat_distance_m_, 0.0, 0.0)));
    geometry_msgs::msg::PoseStamped target_pose;
    target_pose.header.frame_id = planning_frame_;
    target_pose.pose = poseMsgFromSE3(target_world);
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] retreating frame [%s] along local -X by %.3f m before return_home.",
      stage_name.c_str(),
      return_home_j7_retreat_frame_.c_str(),
      return_home_j7_retreat_distance_m_);
    return executeArmPosePlanForFrame(
      target_pose,
      return_home_j7_retreat_frame_,
      true,
      stage_name,
      return_home_j7_retreat_cartesian_max_step_m_);
  }

  bool executeReturnHome(const std::vector<double>& target_joints, const std::string& stage_name) {
    if (armStateMatches(target_joints)) {
      RCLCPP_INFO(this->get_logger(), "Stage [%s] already at target arm joints, skipping planning.", stage_name.c_str());
      return true;
    }
    if (!executeReturnHomeJ7Retreat(stage_name + "_j7_negative_x_retreat")) {
      return false;
    }
    return executeArmJointPlan(target_joints, stage_name, true);
  }

  bool executeFinalMoveViaVerticalDrop(
    const geometry_msgs::msg::PoseStamped& final_target_pose,
    const std::string& final_target_name) {
    const std::string stage_move_above = "move_to_" + final_target_name + "_above";
    const std::string stage_drop = "cartesian_drop_to_" + final_target_name;
    const auto above_pose = offsetAlongBaseZ(final_target_pose, final_approach_height_m_);
    const auto drop_target_pose = offsetAlongBaseZ(above_pose, -final_drop_distance_m_);

    if (!executeArmPosePlan(above_pose, false, stage_move_above)) {
      return false;
    }
    if (!executeArmPosePlan(
          drop_target_pose, true, stage_drop, final_drop_cartesian_max_step_m_)) {
      return false;
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
    req->ee_frame = ee_frame_;
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

    auto resp = callService<lite_motion_msgs::srv::PlanArmMotion>(
      plan_client_, req, "plan_arm_motion");
    if (!resp) {
      return false;
    }
    if (!resp->success) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Arm joint planning failed at stage [%s]: %s",
        stage_name.c_str(),
        resp->message.c_str());
      return false;
    }
    if (resp->trajectory.points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Arm joint planning returned an empty trajectory at stage [%s]", stage_name.c_str());
      return false;
    }

    right_arm_state_ = resp->trajectory.points.back().positions;
    const double wait_sec = effectiveExecutionWaitSec(resp->trajectory);
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] trajectory accepted, waiting %.3f s for execution/playback",
      stage_name.c_str(),
      wait_sec);
    sleepAfterAction(wait_sec);
    return true;
  }

  bool executeHandJointPlan(
    const std::vector<double>& target_joints,
    const std::string& stage_name) {
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
    req->ee_frame = ee_frame_;
    req->use_pose_goal = false;
    req->start_joint_positions = right_hand_state_;
    req->goal_joint_positions = target_joints;
    req->use_cartesian_path = false;
    req->cartesian_max_step = 0.0;

    auto resp = callService<lite_motion_msgs::srv::PlanArmMotion>(
      plan_client_, req, "plan_arm_motion");
    if (!resp) {
      return false;
    }
    if (!resp->success) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Hand planning failed at stage [%s]: %s",
        stage_name.c_str(),
        resp->message.c_str());
      return false;
    }
    if (resp->trajectory.points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Hand planning returned an empty trajectory at stage [%s]", stage_name.c_str());
      return false;
    }

    right_hand_state_ = resp->trajectory.points.back().positions;
    const double wait_sec = effectiveExecutionWaitSec(resp->trajectory);
    RCLCPP_INFO(
      this->get_logger(),
      "Stage [%s] trajectory accepted, waiting %.3f s for execution/playback",
      stage_name.c_str(),
      wait_sec);
    sleepAfterAction(wait_sec);
    return true;
  }

  void sleepAfterAction(double seconds) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (!stop_requested_.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  void bestEffortDetachAndRemoveObject() {
    if (stop_requested_.load()) {
      return;
    }

    clearFinalTargetCollisionAllowanceBestEffort("reset_clear_final_target_e3_cun_allowance");
    clearReturnHomeCollisionAllowanceBestEffort("reset_clear_return_home_e3_cun_allowance");
    clearGraspTargetIgnoreBestEffort("reset_clear_e3_grasp_target_ignore");

    if (object_attached_) {
      if (!setObjectAttached(false)) {
        RCLCPP_WARN(this->get_logger(), "Best-effort detach skipped or failed; continuing reset.");
      }
    }
    if (object_exists_) {
      if (!removeE3Object()) {
        RCLCPP_WARN(this->get_logger(), "Best-effort object removal skipped or failed; continuing reset.");
      }
    }
  }

  bool runSequenceOnce() {
    const auto object_pose = makePoseStamped(p_test_xyz_, p_test_rpy_);
    const auto pregrasp_pose = computePregraspPose();
    geometry_msgs::msg::PoseStamped pregrasp_tcp_pose = pregrasp_pose;
    geometry_msgs::msg::PoseStamped grasp_tcp_pose = object_pose;
    std::string final_target_name;
    const auto final_target_pose = selectedFinalTargetPose(final_target_name);

    if (!addE3Object(object_pose)) {
      return false;
    }
    if (!applyGraspTargetIgnore("before_move_tcp_to_e3_grasp_target")) {
      return false;
    }
    RCLCPP_INFO(
      this->get_logger(),
      "Planning to E3 via pregrasp, then Cartesian approach to p_test. E3 is temporarily ignored only against TCP/J7/finger links during grasp.");
    if (!executeArmPosePlanWithGraspYawSweep(pregrasp_pose, false, "move_tcp_to_e3_pregrasp", &pregrasp_tcp_pose)) {
      clearGraspTargetIgnoreBestEffort("failure_clear_e3_grasp_target_ignore");
      return false;
    }
    grasp_tcp_pose.pose.orientation = pregrasp_tcp_pose.pose.orientation;
    if (!executeArmPosePlan(grasp_tcp_pose, true, "cartesian_approach_tcp_to_e3")) {
      clearGraspTargetIgnoreBestEffort("failure_clear_e3_grasp_target_ignore");
      return false;
    }
    const auto attach_relative_pose = computeAttachRelativePose(object_pose);
    RCLCPP_INFO(
      this->get_logger(),
      "Reached p_test. Entering grasp mode: keep gripper close timing before lift, keep release timing after %s. Bind E3 to tcp with preserved relative pose while allowing tcp/J7/finger touch collisions.",
      final_target_name.c_str());
    if (!setObjectAttached(true, &attach_relative_pose)) {
      clearGraspTargetIgnoreBestEffort("failure_clear_e3_grasp_target_ignore");
      return false;
    }
    clearGraspTargetIgnoreBestEffort("after_attach_clear_e3_grasp_target_ignore");
    if (!executeHandJointPlan(hand_close_positions_, "close_gripper")) {
      return false;
    }
    const auto lift_pose = offsetAlongLocalZ(grasp_tcp_pose, lift_distance_m_);
    if (!executeArmPosePlan(lift_pose, true, "lift_along_e3_positive_z")) {
      return false;
    }
    const std::string move_stage = "move_to_" + final_target_name;
    if (!applyFinalTargetCollisionAllowance("before_" + move_stage + "_e3_cun")) {
      return false;
    }
    RCLCPP_INFO(
      this->get_logger(),
      "Moving to %s via vertical drop: first to target+Z(%.3f m), then Cartesian along base_link -Z(%.3f m). E3-cun collisions are ignored during this phase.",
      final_target_name.c_str(),
      final_approach_height_m_,
      final_drop_distance_m_);
    if (!executeFinalMoveViaVerticalDrop(final_target_pose, final_target_name)) {
      clearFinalTargetCollisionAllowanceBestEffort("failure_clear_final_target_e3_cun_allowance");
      return false;
    }
    const std::string open_stage = "open_gripper_at_" + final_target_name;
    if (!executeHandJointPlan(hand_open_positions_, open_stage)) {
      clearFinalTargetCollisionAllowanceBestEffort("failure_clear_final_target_e3_cun_allowance");
      return false;
    }
    if (!setObjectAttached(false)) {
      clearFinalTargetCollisionAllowanceBestEffort("failure_clear_final_target_e3_cun_allowance");
      return false;
    }
    clearFinalTargetCollisionAllowanceBestEffort("after_release_clear_final_target_e3_cun_allowance");

    if (remove_object_before_return_home_) {
      RCLCPP_WARN(
        this->get_logger(),
        "haomu.remove_object_before_return_home=true is kept for compatibility but ignored by the new flow; E3 will remain in scene until after return_home.");
    }

    if (!applyReturnHomeCollisionAllowance("before_return_home_restore_e3_cun_allowance")) {
      return false;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Post-release return_home: E3 detached from gripper, ignoring E3-base_link/cun collisions during return_home, moving %s to all-zero joint target.",
      right_arm_group_.c_str());
    if (!executeReturnHome(arm_home_positions_, "return_home_after_release")) {
      clearReturnHomeCollisionAllowanceBestEffort("failure_clear_return_home_e3_cun_allowance");
      return false;
    }
    clearReturnHomeCollisionAllowanceBestEffort("after_return_home_clear_e3_cun_allowance");

    if (remove_object_after_release_ || remove_object_before_return_home_) {
      if (!removeE3Object()) {
        return false;
      }
    } else {
      RCLCPP_INFO(
        this->get_logger(),
        "E3 released at %s: detached from TCP and kept as an independent world scene object after return_home.",
        final_target_name.c_str());
    }

    RCLCPP_INFO(this->get_logger(), "haomu_catch sequence completed successfully.");
    return true;
  }

  void runSequence() {
    if (!waitForService<lite_motion_msgs::srv::PlanArmMotion>(plan_client_, "plan_arm_motion") ||
        !waitForService<lite_motion_msgs::srv::AddSceneObject>(add_object_client_, "add_scene_object") ||
        !waitForService<lite_motion_msgs::srv::RemoveSceneObject>(remove_object_client_, "remove_scene_object") ||
        !waitForService<lite_motion_msgs::srv::AttachSceneObject>(attach_object_client_, "attach_scene_object") ||
        !waitForService<lite_motion_msgs::srv::SetSceneCollisionAllowance>(
          set_scene_collision_allowance_client_, "set_scene_collision_allowance")) {
      return;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Starting haomu catch flow. Initial arm state set from haomu.arm_home_positions, hand state set from haomu.hand_open_positions.");

    do {
      bestEffortDetachAndRemoveObject();

      if (!executeHandJointPlan(hand_open_positions_, "reset_open_gripper")) {
        return;
      }
      if (!executeReturnHome(arm_home_positions_, "return_home")) {
        return;
      }
      if (!runSequenceOnce()) {
        return;
      }

      if (sequence_loop_ && !stop_requested_.load()) {
        sleepAfterAction(loop_pause_sec_);
      }
    } while (sequence_loop_ && !stop_requested_.load());
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

  std::string planning_frame_;
  std::string arm_side_{"right"};
  std::string left_arm_group_name_;
  std::string left_hand_group_name_;
  std::string left_tcp_frame_;
  std::string right_arm_group_name_;
  std::string right_hand_group_name_;
  std::string right_tcp_frame_;
  std::string right_arm_group_;
  std::string right_hand_group_;
  std::string ee_frame_;
  std::string object_id_;
  std::string object_mesh_resource_;
  std::vector<double> mesh_scale_;
  std::array<double, 3> p_test_xyz_{};
  std::array<double, 3> p_test_rpy_{};
  double pregrasp_distance_m_{0.04};
  std::string pregrasp_reference_joint_;
  std::vector<std::string> grasp_target_ignore_links_;
  double grasp_target_restore_allowed_penetration_m_{0.005};
  std::array<double, 3> zc_f_xyz_{};
  std::array<double, 3> zc_f_rpy_{};
  std::array<double, 3> zc_m_xyz_{};
  std::array<double, 3> zc_m_rpy_{};
  std::array<double, 3> zc_l_xyz_{};
  std::array<double, 3> zc_l_rpy_{};
  std::array<double, 3> zc_r_xyz_{};
  std::array<double, 3> zc_r_rpy_{};
  std::string final_target_name_{"zc_f"};
  double lift_distance_m_{0.11};
  double cartesian_max_step_m_{0.08};
  bool cartesian_use_sdls_{true};
  double cartesian_nullspace_gain_{0.20};
  double cartesian_singular_threshold_{0.04};
  double cartesian_branch_jump_max_rad_{0.75};
  double cartesian_branch_jump_norm_max_{1.20};
  double cartesian_waypoint_pos_tolerance_{0.0015};
  double cartesian_waypoint_rot_tolerance_{0.020};
  double grasp_yaw_tolerance_rad_{2.0 * M_PI};
  int grasp_yaw_sample_count_{8};
  double final_base_yaw_tolerance_rad_{2.0 * M_PI};
  int final_base_yaw_sample_count_{32};
  double final_approach_height_m_{0.12};
  double final_drop_distance_m_{0.12};
  double final_drop_cartesian_max_step_m_{0.02};
  double home_planning_timeout_{2.0};
  double home_path_check_step_{0.10};
  int home_ompl_max_plan_attempts_{3};
  bool remove_object_before_return_home_{true};
  std::vector<std::string> final_relaxed_object_ids_;
  std::vector<std::string> final_relaxed_links_;
  std::vector<std::string> final_relaxed_links_active_;
  std::vector<std::string> return_home_relaxed_object_ids_;
  std::vector<std::string> return_home_relaxed_links_;
  std::vector<std::string> return_home_relaxed_links_active_;
  double return_home_relaxed_allowed_penetration_m_{0.005};
  std::vector<std::string> return_home_gripper_e3_object_ids_;
  std::vector<std::string> return_home_gripper_e3_links_;
  double return_home_gripper_e3_allowed_penetration_m_{0.002};
  std::string return_home_j7_retreat_frame_;
  double return_home_j7_retreat_distance_m_{0.04};
  double return_home_j7_retreat_cartesian_max_step_m_{0.01};
  bool final_target_collision_allowance_active_{false};
  bool return_home_collision_allowance_active_{false};
  bool return_home_gripper_e3_collision_allowance_active_{false};
  bool grasp_target_ignore_active_{false};
  bool remove_object_after_release_{false};
  double post_action_wait_sec_{0.3};
  double service_wait_sec_{30.0};
  double response_wait_sec_{30.0};
  bool visualization_playback_enabled_{true};
  double visualization_playback_speed_scale_{1.0};
  bool sequence_loop_{true};
  double loop_pause_sec_{0.5};
  bool object_exists_{false};
  bool object_attached_{false};
  std::vector<double> arm_home_positions_;
  std::vector<double> hand_open_positions_;
  std::vector<double> hand_close_positions_;
  std::vector<double> right_arm_state_;
  std::vector<double> right_hand_state_;
};

}  // namespace lite_motion_planner

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<lite_motion_planner::HaomuCatchNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
