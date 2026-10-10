#include "lite_motion_planner/trajectory_generator.hpp"

#include <ruckig/ruckig.hpp>

#include <rclcpp/duration.hpp>

#include <algorithm>
#include <cmath>

namespace lite_motion_planner {

TrajectoryGenerator::TrajectoryGenerator(const RobotModel& robot_model, double control_period_sec)
: robot_model_(robot_model), control_period_sec_(control_period_sec) {}

namespace {

std::vector<Eigen::VectorXd> sparsifyPath(
  const std::vector<Eigen::VectorXd>& path_points,
  double min_segment_norm = 1e-5,
  double max_midpoint_deviation = 5e-4)
{
  if (path_points.size() <= 2) {
    return path_points;
  }

  std::vector<Eigen::VectorXd> out;
  out.reserve(path_points.size());
  out.push_back(path_points.front());

  for (size_t i = 1; i + 1 < path_points.size(); ++i) {
    const auto& prev = out.back();
    const auto& curr = path_points[i];
    const auto& next = path_points[i + 1];

    const double prev_next = (next - prev).norm();
    const double prev_curr = (curr - prev).norm();
    if (prev_curr < min_segment_norm) {
      continue;
    }

    if (prev_next < min_segment_norm) {
      out.push_back(curr);
      continue;
    }

    const double alpha = std::clamp((curr - prev).dot(next - prev) / ((next - prev).squaredNorm() + 1e-12), 0.0, 1.0);
    const Eigen::VectorXd interp = prev + alpha * (next - prev);
    const double deviation = (curr - interp).norm();
    if (deviation > max_midpoint_deviation) {
      out.push_back(curr);
    }
  }

  if ((path_points.back() - out.back()).norm() > 1e-9) {
    out.push_back(path_points.back());
  }

  if (out.size() < 2) {
    out.push_back(path_points.back());
  }
  return out;
}

}  // namespace

trajectory_msgs::msg::JointTrajectory TrajectoryGenerator::generate(
  const std::string& group_name,
  const std::vector<Eigen::VectorXd>& path_points) const
{
  trajectory_msgs::msg::JointTrajectory msg;
  const auto& group = robot_model_.getPlanningGroup(group_name);
  msg.joint_names = group.joint_names;

  const auto sparse_points = sparsifyPath(path_points);

  if (sparse_points.size() < 2) {
    return msg;
  }

  const auto max_vel = robot_model_.velocityLimitsForGroup(group_name);
  const auto max_acc = robot_model_.accelerationLimitsForGroup(group_name);
  const auto max_jerk = robot_model_.jerkLimitsForGroup(group_name);

  const size_t dofs = group.joint_names.size();
  ruckig::Ruckig<ruckig::DynamicDOFs> otg(dofs);

  double accumulated_time = 0.0;
  for (size_t seg = 0; seg + 1 < sparse_points.size(); ++seg) {
    ruckig::InputParameter<ruckig::DynamicDOFs> input(dofs);
    ruckig::Trajectory<ruckig::DynamicDOFs> trajectory(dofs);

    input.current_position.resize(dofs);
    input.current_velocity.resize(dofs, 0.0);
    input.current_acceleration.resize(dofs, 0.0);
    input.target_position.resize(dofs);
    input.target_velocity.resize(dofs, 0.0);
    input.target_acceleration.resize(dofs, 0.0);
    input.max_velocity.resize(dofs);
    input.max_acceleration.resize(dofs);
    input.max_jerk.resize(dofs);

    for (size_t i = 0; i < dofs; ++i) {
      input.current_position[i] = sparse_points[seg][static_cast<Eigen::Index>(i)];
      input.target_position[i] = sparse_points[seg + 1][static_cast<Eigen::Index>(i)];
      input.max_velocity[i] = max_vel[static_cast<Eigen::Index>(i)];
      input.max_acceleration[i] = max_acc[static_cast<Eigen::Index>(i)];
      input.max_jerk[i] = max_jerk[static_cast<Eigen::Index>(i)];
    }

    const double delta_norm = (sparse_points[seg + 1] - sparse_points[seg]).norm();
    if (delta_norm < 1e-6) {
      continue;
    }

    const auto result = otg.calculate(input, trajectory);
    if (result < 0) {
      continue;
    }

    const bool skip_first_sample = !msg.points.empty();
    for (double t = 0.0; t <= trajectory.get_duration() + 1e-9; t += control_period_sec_) {
      if (skip_first_sample && t < 1e-12) {
        continue;
      }
      std::vector<double> pos(dofs), vel(dofs), acc(dofs);
      trajectory.at_time(t, pos, vel, acc);

      trajectory_msgs::msg::JointTrajectoryPoint pt;
      pt.positions = pos;
      pt.velocities = vel;
      pt.accelerations = acc;
      pt.time_from_start = rclcpp::Duration::from_seconds(accumulated_time + t);
      msg.points.push_back(std::move(pt));
    }

    accumulated_time += trajectory.get_duration();
  }

  return msg;
}

}  // namespace lite_motion_planner
