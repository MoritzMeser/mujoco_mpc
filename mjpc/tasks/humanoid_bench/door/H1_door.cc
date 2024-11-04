//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_door.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench door task ---------------- //
// ----------------------------------------------------------------------------
// //
void H1_door::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                   double *residual) const {
  // ----- set parameters ----- //
  double const standHeight = 1.65;
  double direction_goal = 0.0;
  double speed_goal;
  if (task_->reward_machine_state_ <= 2) {
    speed_goal = 0.0;
  } else {  // state 3
    speed_goal = 1.0;
  }

  // ----- standing ----- //
  double head_height = SensorByName(model, data, "head_height")[2];
  double standing_hb =
      tolerance(head_height, {standHeight, INFINITY}, standHeight / 4);

  // ----- torso upright ----- //
  double torso_upright = SensorByName(model, data, "torso_upright")[2];
  double upright =
      tolerance(torso_upright, {0.9, INFINITY}, 0.9, "linear", 0.0);

  double stand_reward = standing_hb * upright;

  // ----- small control ----- //
  double small_control = 0.0;
  for (int i = 0; i < model->nu; i++) {
    small_control +=
        tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
  }
  small_control /= model->nu;  // average over all controls
  small_control = (4 + small_control) / 5;

  // ----- door openness reward ----- //
  double door_openness_reward =
      std::min(1.0, (data->qpos[model->nq - 2] / 1) *
                        std::abs(data->qpos[model->nq - 2] / 1));

  // ----- door hatch openness reward ----- //
  double door_hatch_openness_reward =
      tolerance(data->qpos[model->nq - 1], {0.75, 2}, 0.75, "linear");

  // ----- hand hatch proximity reward ----- //
  double *door_hatch_pos = SensorByName(model, data, "door_hatch");
  double *left_hand_pos = SensorByName(model, data, "left_hand_position");
  double *right_hand_pos = SensorByName(model, data, "right_hand_position");

  double left_hand_hatch_closeness =
      std::sqrt(std::pow(door_hatch_pos[0] - left_hand_pos[0], 2) +
                std::pow(door_hatch_pos[1] - left_hand_pos[1], 2) +
                std::pow(door_hatch_pos[2] - left_hand_pos[2], 2));
  double right_hand_hatch_closeness =
      std::sqrt(std::pow(door_hatch_pos[0] - right_hand_pos[0], 2) +
                std::pow(door_hatch_pos[1] - right_hand_pos[1], 2) +
                std::pow(door_hatch_pos[2] - right_hand_pos[2], 2));

  double hand_hatch_proximity_reward =
      tolerance(std::min(right_hand_hatch_closeness, left_hand_hatch_closeness),
                {0, 0.25}, 1, "linear");

  // ----- passage reward ----- //
  double passage_reward = tolerance(SensorByName(model, data, "imu")[0],
                                    {1.2, INFINITY}, 1, "linear", 0.0);

  // ----- total reward ----- //
  double reward = 0.1 * stand_reward * small_control +
                  0.45 * door_openness_reward +
                  0.05 * door_hatch_openness_reward +
                  0.05 * hand_hatch_proximity_reward + 0.35 * passage_reward;

  // ----- residuals ----- //

  residual[0] = std::exp(-reward);

  //
  //
  //
  int counter = 1;

  double torso_height_goal = 1.25;
  // ----- torso height ----- //
  double torso_height = SensorByName(model, data, "torso_position")[2];
  residual[counter++] = torso_height - torso_height_goal;

  // ----- pelvis / feet ----- //
  double *foot_right = SensorByName(model, data, "foot_right");
  double *foot_left = SensorByName(model, data, "foot_left");
  double pelvis_height = SensorByName(model, data, "pelvis_position")[2];
  residual[counter++] =
      0.5 * (foot_left[2] + foot_right[2]) - pelvis_height - 0.2;

  // ----- balance ----- //
  // capture point
  double *subcom = SensorByName(model, data, "torso_subcom");
  double *subcomvel = SensorByName(model, data, "torso_subcomvel");

  double capture_point[3];
  mju_addScl(capture_point, subcom, subcomvel, 0.3, 3);
  capture_point[2] = 1.0e-3;

  // project onto line segment

  double axis[3];
  double center[3];
  double vec[3];
  double pcp[3];
  mju_sub3(axis, foot_right, foot_left);
  axis[2] = 1.0e-3;
  double length = 0.5 * mju_normalize3(axis) - 0.05;
  mju_add3(center, foot_right, foot_left);
  mju_scl3(center, center, 0.5);
  mju_sub3(vec, capture_point, center);

  // project onto axis
  double t = mju_dot3(vec, axis);

  // clamp
  t = mju_max(-length, mju_min(length, t));
  mju_scl3(vec, axis, t);
  mju_add3(pcp, vec, center);
  pcp[2] = 1.0e-3;

  // is standing
  double standing =
      torso_height / mju_sqrt(torso_height * torso_height + 0.45 * 0.45) - 0.4;

  standing = std::max(0.0, standing);

  mju_sub(&residual[counter], capture_point, pcp, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);

  counter += 2;

  // ----- upright ----- //
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

  // ----- posture ----- //
  mju_sub(&residual[counter], data->qpos + 7, model->key_qpos + 7, model->nu);
  counter += model->nu;

  // ----- Walk ----- //
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

  // Face in right-direction
  double direction_goal_radiant = direction_goal * M_PI / 180;
  double face_x[2] = {cos(direction_goal_radiant), sin(direction_goal_radiant)};
  mju_sub(&residual[counter], forward, face_x, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);
  counter += 2;

  // com vel
  double *waist_lower_subcomvel =
      SensorByName(model, data, "waist_lower_subcomvel");
  double *torso_velocity = SensorByName(model, data, "torso_velocity");
  double com_vel[2];
  mju_add(com_vel, waist_lower_subcomvel, torso_velocity, 2);
  mju_scl(com_vel, com_vel, 0.5, 2);

  // Walk forward
  residual[counter++] = standing * (mju_dot(com_vel, forward, 2) - speed_goal);

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

  // ----- control ----- //
  mju_sub(&residual[counter], data->ctrl, model->key_qpos + 7,
          model->nu);  // because of pos control
  counter += model->nu;

  // door open
  residual[counter++] = data->qpos[model->nq - 2] - 1.4;

  if (task_->reward_machine_state_ == 1) {
    // right hand distance
    double *right_hand_position =
        SensorByName(model, data, "right_hand_position");
    double *door_hatch = SensorByName(model, data, "door_hatch_inside");
    mju_sub3(&residual[counter], right_hand_position, door_hatch);
    counter += 3;
  } else if (task_->reward_machine_state_ == 2) {
    // right hand distance
    double *right_hand_position =
        SensorByName(model, data, "right_hand_position");
    double *door_hatch = SensorByName(model, data, "door_hatch_outside");
    mju_sub3(&residual[counter], right_hand_position, door_hatch);
    counter += 3;
  } else {
    mju_scl3(&residual[counter], right_hand_pos, 0.0);  // zero residual
    counter += 3;
  }

  if (task_->reward_machine_state_ == 3) {
    // unwanted collisions
    int floor_idx = mj_name2id(model, mjOBJ_GEOM, "floor");

    int unwanted_collision = 0;
    for (int i = 0; i < data->ncon; i++) {
      if (data->contact[i].geom1 != floor_idx &&
          data->contact[i].geom2 != floor_idx) {
        unwanted_collision++;
      }
    }
    residual[counter++] = unwanted_collision;
  } else {
    residual[counter++] = 0.0;
  }

  // robot position
  double x_displacement = data->qpos[model->nq - 2] / 3.0;
  if (task_->reward_machine_state_ <= 2) {
    if (data->qpos[0] < -x_displacement) {
      residual[counter++] = 0.0;
    } else {
      residual[counter++] = data->qpos[0] + x_displacement;
    }
    if (data->qpos[1] > 0.25) {
      residual[counter++] = 0.0;
    } else {
      residual[counter++] = -data->qpos[1] + 0.25   ;
    }
  } else {
    residual[counter++] = 2.0 - data->qpos[0];
    residual[counter++] = 0.0;
  }

  // close door penalty
  residual[counter++] = std::abs(std::min(data->qvel[model->nv - 2], 0.0));

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

// -------- Transition for humanoid_bench door task -------- //
// ------------------------------------------------------------ //
void H1_door::TransitionLocked(mjModel *model, mjData *data) {
  //
  if (reward_machine_state_ == 1) {
    // check if door is open
    if (data->qpos[model->nq - 2] > 0.5) {
      reward_machine_state_ = 2;
      printf("Door is half open - switch to state 2\n");
    }
  } else if (reward_machine_state_ == 2) {
    if (data->qpos[model->nq - 2] > 1.3) {
      reward_machine_state_ = 3;
      printf("Door is full open - switch to state 3\n");
    }
  }
}

}  // namespace mjpc