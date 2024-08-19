
#include "H1_package.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include "mjpc/tasks/humanoid_bench/utility/utility_functions.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench package task ----------------
// //
// -------------------------------------------------------------------------------
// //
void H1_package::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                      double *residual) const {
  // ----- set parameters ----- //
  double const stand_height = 1.65;

  // compute walk speed
  double walk_speed = 0.0;
  if (task_->reward_machine_state_ == 0) {
    walk_speed = 0.0;
  } else if (task_->reward_machine_state_ == 1) {
    walk_speed = 1.0;
  }

  // compute walk direction
  double walk_direction[2];
  if (task_->reward_machine_state_ == 0) {
    walk_direction[0] = 1.0;
    walk_direction[1] = 0.0;
  } else if (task_->reward_machine_state_ == 1) {
    mju_sub(walk_direction, task_->target_position_.data(),
            SensorByName(model, data, "torso_position"), 2);
    mju_normalize(walk_direction, 2);
  }

  // ----- standing ----- //
  double head_height = SensorByName(model, data, "head_height")[2];
  double hb_standing =
      tolerance(head_height, {stand_height, INFINITY}, stand_height / 4);

  // ----- torso upright ----- //
  double torso_upright = SensorByName(model, data, "torso_upright")[2];
  double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9);

  double stand_reward = hb_standing * upright;

  // ----- small control ----- //
  double small_control = 0.0;
  for (int i = 0; i < model->nu; i++) {
    small_control +=
        tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
  }
  small_control /= model->nu;  // average over all controls
  small_control = (4 + small_control) / 5;

  // ----- rewards specific to the package task ----- //

  double *package_location = SensorByName(model, data, "package_location");
  double const *package_destination = task_->target_position_.data();
  double *left_hand_location = SensorByName(model, data, "left_hand_pos");
  double *right_hand_location = SensorByName(model, data, "right_hand_pos");

  double dist_package_destination =
      std::hypot(std::hypot(package_location[0] - package_destination[0],
                            package_location[1] - package_destination[1]),
                 package_location[2] - package_destination[2]);

  double dist_hand_package_right =
      std::hypot(std::hypot(right_hand_location[0] - package_location[0],
                            right_hand_location[1] - package_location[1]),
                 right_hand_location[2] - package_location[2]);

  double dist_hand_package_left =
      std::hypot(std::hypot(left_hand_location[0] - package_location[0],
                            left_hand_location[1] - package_location[1]),
                 left_hand_location[2] - package_location[2]);

  double package_height = std::min(package_location[2], 1.0);

  bool reward_success = dist_package_destination < 0.1;

  // ----- reward computation ----- //
  double reward = (stand_reward * small_control - 3 * dist_package_destination -
                   (dist_hand_package_left + dist_hand_package_right) * 0.1 +
                   package_height + reward_success * 1000);

  // ----- residuals ----- //
  int counter = 0;
  residual[counter++] = std::exp(-reward);
  // ----- torso height ----- //
  double torso_height = SensorByName(model, data, "torso_position")[2];
  residual[counter++] = torso_height - parameters_[0];

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
  if (task_->reward_machine_state_ == 1) {
    mju_sub(&residual[counter], forward, walk_direction, 2);
    mju_scl(&residual[counter], &residual[counter], standing, 2);
    counter += 2;
  } else {
    residual[counter++] = 0;
    residual[counter++] = 0;
  }

  // com vel
  double *waist_lower_subcomvel =
      SensorByName(model, data, "waist_lower_subcomvel");
  double *torso_velocity = SensorByName(model, data, "torso_velocity");
  double com_vel[2];
  mju_add(com_vel, waist_lower_subcomvel, torso_velocity, 2);
  mju_scl(com_vel, com_vel, 0.5, 2);

  // Walk forward
  residual[counter++] = standing * (mju_dot(com_vel, forward, 2) - walk_speed);

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

  // ----- right hand distance ----- //
  if (task_->reward_machine_state_ == 0 || task_->reward_machine_state_ == 1) {
    mju_sub3(&residual[counter], right_hand_location, package_location);
    mju_scl3(&residual[counter], &residual[counter], standing);
    counter += 3;
  } else {
    mju_zero3(&residual[counter]);
    counter += 3;
  }

  // ----- left hand distance ----- //
  if (task_->reward_machine_state_ == 0 || task_->reward_machine_state_ == 1) {
    mju_sub3(&residual[counter], left_hand_location, package_location);
    mju_scl3(&residual[counter], &residual[counter], standing);
    counter += 3;
  } else {
    mju_zero3(&residual[counter]);
    counter += 3;
  }

  // ----- goal distance ----- //
  if (task_->reward_machine_state_ == 2) {
    mju_sub3(&residual[counter], package_location, package_destination);
    mju_scl3(&residual[counter], &residual[counter], standing);
    counter += 3;
  } else {
    mju_zero3(&residual[counter]);
    counter += 3;
  }

  // package height
  //  0.55 0 0.35
  residual[counter++] = 0;  // package_location[0] - 0.55;
  residual[counter++] = 0;  // package_location[1] - 0.0;
  residual[counter++] = package_location[2] - 0.5;

  // contact right hand
  if (task_->reward_machine_state_ == 0 || task_->reward_machine_state_ == 1) {
    bool contact_right_hand = CheckBodyCollision(
        model, data, "right_elbow_collision", "package_a_collision");

    if (contact_right_hand) {
      residual[counter++] = 0;
    } else {
      residual[counter++] = 1;
    }
  } else {
    residual[counter++] = 0;
  }

  // contact left hand
  if (task_->reward_machine_state_ == 0 || task_->reward_machine_state_ == 1) {
    bool contact_left_hand = CheckBodyCollision(
        model, data, "left_elbow_collision", "package_a_collision");

    if (contact_left_hand) {
      residual[counter++] = 0;
    } else {
      residual[counter++] = 1;
    }
  } else {
    residual[counter++] = 0;
  }

  // no contact with the ground
  if (task_->reward_machine_state_ == 0 || task_->reward_machine_state_ == 1) {
    bool contact_ground =
        CheckBodyCollision(model, data, "floor", "package_a_collision");

    if (contact_ground) {
      residual[counter++] = 1;
    } else {
      residual[counter++] = 0;
    }
  } else {
    residual[counter++] = 0;
  }

  // foot on the ground
  bool contact_foot_right =
      CheckBodyCollision(model, data, "floor", "right_foot_collision");
  bool contact_foot_left =
      CheckBodyCollision(model, data, "floor", "left_foot_collision");

  if (contact_foot_right) {
    residual[counter++] = 0;
  } else {
    residual[counter++] = 1;
  }

  if (contact_foot_left) {
    residual[counter++] = 0;
  } else {
    residual[counter++] = 1;
  }

  // unwanted collisions
  std::vector<int> exception_list;
  exception_list.push_back(
      mj_name2id(model, mjOBJ_GEOM, "right_elbow_collision"));
  exception_list.push_back(
      mj_name2id(model, mjOBJ_GEOM, "left_elbow_collision"));
  exception_list.push_back(mj_name2id(model, mjOBJ_GEOM, "floor"));

  int package_index = mj_name2id(model, mjOBJ_GEOM, "package_a_collision");

  int unwanted_collision = 0;
  for (int i = 0; i < data->ncon; i++) {
    if ((data->contact[i].geom1 == package_index &&
         std::find(exception_list.begin(), exception_list.end(),
                   data->contact[i].geom2) == exception_list.end()) ||
        (data->contact[i].geom2 == package_index &&
         std::find(exception_list.begin(), exception_list.end(),
                   data->contact[i].geom1) == exception_list.end())) {
      unwanted_collision++;
    }
  }

  residual[counter++] = unwanted_collision;

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

// -------- Transition for humanoid_bench package task --------
// for a more complex task this might be necessary (like packageing to different
// targets)
// ---------------------------------------------
void H1_package::TransitionLocked(mjModel *model, mjData *data) {
  // show the target position
  mju_copy3(data->mocap_pos, target_position_.data());

  // update state machine
  if (reward_machine_state_ == 0) {  // lift state
    bool switch_to_walk = true;

    // check package height
    double *package_location = SensorByName(model, data, "package_location");
    if (package_location[2] < 1.0) {
      switch_to_walk = false;
    }

    // check if package is gripped
    bool contact_right_hand = CheckBodyCollision(
        model, data, "right_elbow_collision", "package_a_collision");
    bool contact_left_hand = CheckBodyCollision(
        model, data, "left_elbow_collision", "package_a_collision");
    if (!contact_right_hand || !contact_left_hand) {
      switch_to_walk = false;
    }

    // check if package is not fast moving
    double *package_velocity = SensorByName(model, data, "package_velocity");
    double package_speed = mju_norm3(package_velocity);
    if (package_speed > 0.2) {  // ToDo: fine tune this value
      switch_to_walk = false;
    }

    // ToDo: check if robot is stable

    if (switch_to_walk) {
      reward_machine_state_ = 1;
      printf("Switching to walk state\n");
    }
  } else if (reward_machine_state_ == 1) {  // walk state
    bool switch_to_place = true;

    // check distance to target
    double *package_location = SensorByName(model, data, "package_location");
    double dist_package_destination =
        std::sqrt(std::pow(package_location[0] - target_position_[0], 2) +
                  std::pow(package_location[1] - target_position_[1], 2));
    if (dist_package_destination > 0.6) {
      switch_to_place = false;
    }

    if (switch_to_place) {
      reward_machine_state_ = 2;
      printf("Switching to place state\n");
    }
  }
}

void H1_package::ResetLocked(const mjModel *model) {
  //  std::random_device rd;
  //  std::mt19937 gen(rd());
  //  std::uniform_real_distribution<> dis_x(-2, 2);
  //  std::uniform_real_distribution<> dis_y(-2, 2);
  //
  //  target_position_ = {dis_x(gen), dis_y(gen), 0.35};
  //  printf("Initial target position: %f, %f\n", target_position_[0],
  //         target_position_[1]);
}
}  // namespace mjpc
