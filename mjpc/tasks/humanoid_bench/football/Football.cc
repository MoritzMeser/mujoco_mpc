//
// Created by Moritz Meser on 21.05.24.
//

#include "Football.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench basketball task
// ---------------- //
// ----------------------------------------------------------------------------------
// //
void Football::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                    double *residual) const {
  double const height_goal = parameters_[0];
  double const walk_speed = parameters_[1];
  double const rm_state = task_->reward_machine_state_;

  // read sensors
  double torso_height = SensorByName(model, data, "torso_position")[2];
  double *foot_right_pos = SensorByName(model, data, "foot_right_pos");
  double *foot_left_pos = SensorByName(model, data, "foot_left_pos");

  // is standing
  double standing =
      torso_height / mju_sqrt(torso_height * torso_height + 0.45 * 0.45) - 0.4;

  // compute forward
  double *torso_forward = SensorByName(model, data, "torso_forward");
  double *pelvis_forward = SensorByName(model, data, "pelvis_forward");
  double *foot_right_forward = SensorByName(model, data, "foot_right_forward");
  double *foot_left_forward = SensorByName(model, data, "foot_left_forward");

  double forward[2];
  mju_copy(forward, torso_forward, 2);
  mju_addTo(forward, pelvis_forward, 2);
  mju_addTo(forward, foot_right_forward, 2);
  mju_addTo(forward, foot_left_forward, 2);
  mju_normalize(forward, 2);

  // compute com velocity
  double *waist_lower_subcomvel =
      SensorByName(model, data, "waist_lower_subcomvel");
  double *torso_velocity = SensorByName(model, data, "torso_velocity");
  double com_vel[2];
  mju_add(com_vel, waist_lower_subcomvel, torso_velocity, 2);
  mju_scl(com_vel, com_vel, 0.5, 2);

  // init counter
  int counter = 0;

  // head-feet-error
  if (rm_state == 0 || rm_state == 1 || rm_state == 2 || rm_state == 3) {
    double *head_position = SensorByName(model, data, "head_position");
    double head_feet_error =
        head_position[2] - 0.5 * (foot_right_pos[2] + foot_left_pos[2]);
    residual[counter++] = head_feet_error - height_goal;
  } else {
    residual[counter++] = 0.0;
  }

  // balance
  if (rm_state == 0 || rm_state == 1 || rm_state == 2 || rm_state == 3) {
    // capture point
    double *subcom =
        SensorByName(model, data, "torso_subcom");  // TODO: redundant
    double *subcomvel = SensorByName(model, data, "torso_subcomvel");

    double capture_point[3];
    mju_addScl(capture_point, subcom, subcomvel, 0.3, 3);
    capture_point[2] = 1.0e-3;

    // project onto line segment

    double axis[3];
    double center[3];
    double vec[3];
    double pcp[3];
    mju_sub3(axis, foot_right_pos, foot_left_pos);
    axis[2] = 1.0e-3;
    double length = 0.5 * mju_normalize3(axis) - 0.05;
    mju_add3(center, foot_right_pos, foot_left_pos);
    mju_scl3(center, center, 0.5);
    mju_sub3(vec, capture_point, center);

    // project onto axis
    double t = mju_dot3(vec, axis);

    // clamp
    t = mju_max(-length, mju_min(length, t));
    mju_scl3(vec, axis, t);
    mju_add3(pcp, vec, center);
    pcp[2] = 1.0e-3;

    standing = std::max(0.0, standing);

    mju_sub(&residual[counter], capture_point, pcp, 2);
    mju_scl(&residual[counter], &residual[counter], standing, 2);

    counter += 2;
  } else {
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
  }

  // upright
  if (rm_state == 0 || rm_state == 1 || rm_state == 2 || rm_state == 3) {
    double *torso_up = SensorByName(model, data, "torso_up");
    double *pelvis_up = SensorByName(model, data, "pelvis_up");
    double *foot_right_up = SensorByName(model, data, "foot_right_up");
    double *foot_left_up = SensorByName(model, data, "foot_left_up");
    double z_ref[3] = {0.0, 0.0, 1.0};

    // torso
    residual[counter++] = torso_up[2] - 1.0;

    // pelvis
    residual[counter++] = 0.3 * (pelvis_up[2] - 1.0);

    // right foot
    mju_sub3(&residual[counter], foot_right_up, z_ref);
    mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
    counter += 3;

    mju_sub3(&residual[counter], foot_left_up, z_ref);
    mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
    counter += 3;
  } else {
    mju_scl(&residual[counter], &residual[counter], 0.0, 8);
    counter += 8;
  }

  // posture
  if (rm_state == 0 || rm_state == 1 || rm_state == 2 || rm_state == 3) {
    mju_sub(&residual[counter], data->qpos + 7, model->key_qpos + 7, model->nu);
    counter += model->nu;
  } else {
    mju_scl(&residual[counter], &residual[counter], 0.0, model->nu);
    counter += model->nu;
  }

  // face direction
  if (rm_state == 0 || rm_state == 1 || rm_state == 2 || rm_state == 3) {
    mju_sub(&residual[counter], forward, task_->robot_facing_dir_.data(), 2);
    mju_scl(&residual[counter], &residual[counter], standing, 2);
    counter += 2;
  } else {
    mju_scl(&residual[counter], &residual[counter], 0.0, 2);
    counter += 2;
  }

  // walk
  if (rm_state == 1) {
    // Walk Torso  + Walk Feet
    residual[counter++] =
        standing * (mju_dot(com_vel, forward, 2) - walk_speed);

    // ----- move feet ----- //
    double *foot_right_vel = SensorByName(model, data, "foot_right_velocity");
    double *foot_left_vel = SensorByName(model, data, "foot_left_velocity");
    double move_feet[2];
    mju_copy(move_feet, com_vel, 2);
    mju_addToScl(move_feet, foot_right_vel, -0.5, 2);
    mju_addToScl(move_feet, foot_left_vel, -0.5, 2);

    mju_copy(&residual[counter], move_feet, 2);
    mju_scl(&residual[counter], &residual[counter], standing, 2);
    counter += 2;
  } else {
    // do not walk
    residual[counter++] = standing * (mju_dot(com_vel, forward, 2));
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
  }

  // control
  if (rm_state == 0 || rm_state == 1 || rm_state == 2 || rm_state == 3) {
    mju_sub(&residual[counter], data->ctrl, model->key_qpos + 7, model->nu);
    counter += model->nu;
  } else {
    mju_scl(&residual[counter], &residual[counter], 0.0, model->nu);
    counter += model->nu;
  }

  if (rm_state == 2) {
    double *ball_position = SensorByName(model, data, "football_position");
    double *ball_velocity = SensorByName(model, data, "football_velocity");

    // ball position
    mju_sub3(&residual[counter], task_->target_position_.data(), ball_position);
    mju_scl3(&residual[counter], &residual[counter], standing);
    counter += 3;

    // ball velocity
    double ball_goal_movement[2];
    mju_sub(ball_goal_movement, task_->target_position_.data(), ball_position,
            2);
    mju_normalize(ball_goal_movement, 2);
    mju_sub(&residual[counter], ball_velocity, ball_goal_movement, 2);
    mju_scl(&residual[counter], &residual[counter], standing, 2);
    counter += 2;
  } else {
    mju_scl(&residual[counter], &residual[counter], 0.0, 5);
    counter += 5;
  }

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }
}

// -------- Transition for humanoid_bench basketball task -------- //
// ------------------------------------------------------------ //
void Football::TransitionLocked(mjModel *model, mjData *data) {
  // set mocap
  mju_copy3(data->mocap_pos, target_position_.data());
  mju_copy3(data->mocap_pos + 3, robot_goal_position_.data());

  // state machine
  if (reward_machine_state_ == 0) {
    // compute distance robot to robot_goal_position_
    double *torso_position = SensorByName(model, data, "torso_position");
    double distance =
        std::sqrt(std::pow(torso_position[0] - robot_goal_position_[0], 2) +
                  std::pow(torso_position[1] - robot_goal_position_[1], 2));
    if (distance > 0.2) {
      reward_machine_state_ = 1;  // switch to walk
      printf("switch to walk state\n");
    } else {
      // set facing direction to ball goal
      mju_sub(robot_facing_dir_.data(), target_position_.data(), torso_position,
              2);
      // normalize facing direction vector
      mju_normalize(robot_facing_dir_.data(), 2);

      // switch to shoot
      reward_machine_state_ = 2;
      printf("switch to shoot state\n");
    }
    // TODO: switch to shoot
  }
  if (reward_machine_state_ == 1) {
    // compute distance robot to robot_goal_position_
    double *torso_position = SensorByName(model, data, "torso_position");
    double distance =
        std::sqrt(std::pow(torso_position[0] - robot_goal_position_[0], 2) +
                  std::pow(torso_position[1] - robot_goal_position_[1], 2));
    if (distance < 0.2) {
      reward_machine_state_ = 0;  // switch to stand
      printf("switch to stand state\n");
    } else {
      // set facing direction as direction to stand goal
      mju_sub(robot_facing_dir_.data(), robot_goal_position_.data(),
              torso_position, 2);
      // normalize facing direction vector
      mju_normalize(robot_facing_dir_.data(), 2);
    }
  }
  if (reward_machine_state_ == 2) {
    if (mju_norm3(SensorByName(model, data, "football_velocity")) > 0.5) {
      reward_machine_state_ = 3;
      printf("switch to wait state\n");
    }
  }
  if (reward_machine_state_ == 3) {
    if (mju_norm3(SensorByName(model, data, "football_velocity")) < 0.4) {
      reward_machine_state_ = 0;
      printf("switch to stand state\n");
    }
  }
}

}  // namespace mjpc
