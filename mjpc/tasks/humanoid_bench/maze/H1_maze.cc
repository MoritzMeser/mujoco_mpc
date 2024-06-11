//
// Created by Moritz Meser on 20.05.24.
//

#include "H1_maze.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include "mjpc/tasks/humanoid_bench/utility/utility_functions.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench maze task ---------------- //
// ---------------------------------------------------------------------------- //
    void H1_maze::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                       double *residual) const {
//        // ----- set parameters ----- //
//        double const standHeight = 1.65;
//        double const moveSpeed = 2.0;
//
//
//        // ----- standing ----- //
//        double head_height = SensorByName(model, data, "head_height")[2];
//        double standing = tolerance(head_height, {standHeight, INFINITY}, standHeight / 4);
//
//
//        // ----- torso upright ----- //
//        double torso_upright = SensorByName(model, data, "torso_upright")[2];
//        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9);
//
//        double standReward = standing * upright;
//
//
//        // ----- small control ----- //
//        double small_control = 0.0;
//        for (int i = 0; i < model->nu; i++) {
//            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
//        }
//        small_control /= model->nu;  // average over all controls
//        small_control = (4 + small_control) / 5;
//
//        // ----- wall collision ----- //
//        double wall_collision_discount = 1.0;
//        int maze_id = mj_name2id(model, mjOBJ_BODY, "maze");
//
//        // Iterate over all the child bodies of the "maze" body
//        for (int i = 0; i < model->nbody; i++) {
//            if (model->body_parentid[i] == maze_id) {
//                // Get the ID of the child body
//                int child_body_id = i;
//
//                // Iterate over all the geometries
//                for (int j = 0; j < model->ngeom; j++) {
//                    if (model->geom_bodyid[j] == child_body_id) {
//                        // Get the ID of the geometry
//                        int geom_id = j;
//
//                        // Check for collisions
//                        if (CheckAnyCollision(model, data, geom_id)) {
//                            wall_collision_discount = 0.1;
//                            break;
//                        }
//                    }
//                }
//            }
//        }
//
//
//        // ----- stage convert reward ----- //
//        double stage_convert_reward = 0.0;
//        double *goal_pos = data->mocap_pos;
//        double *pelvis_pos = SensorByName(model, data, "pelvis_position");
//        double dist = std::sqrt(std::pow(goal_pos[0] - pelvis_pos[0], 2) +
//                                std::pow(goal_pos[1] - pelvis_pos[1], 2) +
//                                std::pow(goal_pos[2] - pelvis_pos[2], 2));
//
//        // check if task is done
//        if (dist < 0.1) {
//            stage_convert_reward = 100.0;
//        }
//
//        // ----- move speed ----- //
//        double move_direction[3] = {0, 0, 0};
//
//        // Check each case
//        if (task_->curr_goal_idx_ == 0) {
//            // checkpoint is 0,0,1
//            move_direction[0] = 1;
//            move_direction[1] = 0;
//            move_direction[2] = 0;
//        } else if (task_->curr_goal_idx_ == 1) {
//           // checkpoint is 3,0,1
//            move_direction[0] = 1;
//            move_direction[1] = 0;
//            move_direction[2] = 0;
//        } else if (task_->curr_goal_idx_ == 2) {
//            // checkpoint is 3,6,1
//            move_direction[0] = 0;
//            move_direction[1] = 1;
//            move_direction[2] = 0;
//        } else if (task_->curr_goal_idx_ == 3) {
//            // checkpoint is 6,6,1
//            move_direction[0] = 1;
//            move_direction[1] = 0;
//            move_direction[2] = 0;
//        } else {
//            // last checkpoint is reached
//            move_direction[0] = 0;
//            move_direction[1] = 0;
//            move_direction[2] = 0;
//        }
//
//        // Get the center of mass velocity
//        double *com_velocity = SensorByName(model, data, "center_of_mass_velocity");
//
//        double move;
//        if (task_->curr_goal_idx_ == 4) {
//            // last checkpoint is reached
//            move = 1.0;
//        } else {
//            // Calculate the move reward
//            move = tolerance(com_velocity[0] - move_direction[0] * moveSpeed, {0, 0}, 1.0, "linear", 0.0) *
//                   tolerance(com_velocity[1] - move_direction[1] * moveSpeed, {0, 0}, 1.0, "linear", 0.0);
//        }
//        move = (5 * move + 1) / 6;
//        // ----- checkpoint proximity ----- //
//        double checkpoint_proximity = std::sqrt(std::pow(goal_pos[0] - pelvis_pos[0], 2) +
//                                                std::pow(goal_pos[1] - pelvis_pos[1], 2));
//        double checkpoint_proximity_reward = tolerance(checkpoint_proximity, {0, 0.0}, 1.0);
//
//        // ----- reward ----- //
//        double reward = (0.2 * (standReward * small_control)
//                         + 0.4 * move
//                         + 0.4 * checkpoint_proximity_reward
//                        ) * wall_collision_discount + stage_convert_reward;
//
//        // ----- residuals ----- //
//        residual[0] = std::exp(-reward);

        int counter = 0;

        // ----- torso height ----- //
        double head_height = SensorByName(model, data, "torso_position")[2];
        residual[counter++] = head_height - parameters_[0];

        // ----- pelvis / feet ----- //
        double *foot_right = SensorByName(model, data, "foot_right");
        double *foot_left = SensorByName(model, data, "foot_left");
        double pelvis_height = SensorByName(model, data, "pelvis_position")[2];
        residual[counter++] =
                0.5 * (foot_left[2] + foot_right[2]) - pelvis_height - parameters_[2];

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
        double torso_height = SensorByName(model, data, "torso_position")[2];
        double standing =
                torso_height / mju_sqrt(torso_height * torso_height + 0.45 * 0.45) - 0.4;

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
        mju_copy(&residual[counter], data->qpos + 7, model->nq - 7);
        counter += model->nq - 7;

        // ----- walk ----- //
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

        // com vel
        double *waist_lower_subcomvel =
                SensorByName(model, data, "waist_lower_subcomvel");
        double *torso_velocity = SensorByName(model, data, "torso_velocity");
        double com_vel[2];
        mju_add(com_vel, waist_lower_subcomvel, torso_velocity, 2);
        mju_scl(com_vel, com_vel, 0.5, 2);

        // walk forward
        residual[counter++] =
                standing * (mju_dot(com_vel, forward, 2) - parameters_[1]);

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
        mju_copy(&residual[counter], data->ctrl, model->nu);
        counter += model->nu;

        // ------ goal dist ------ //
        double *goal_pos = model->key_mpos + 3 * (task_->curr_goal_idx_ + 1);  // offset 1 is on purpose
        double *pelvis_pos = SensorByName(model, data, "pelvis_position");
        mju_sub(&residual[counter], goal_pos, pelvis_pos, 2);
        mju_scl(&residual[counter], &residual[counter], standing, 2);
        counter += 2;

//        // ----- orientation ----- //
        double *pelvis_x_axis = SensorByName(model, data, "pelvis_x_axis");
        mju_normalize(pelvis_x_axis, 2);

        // calculate direction vector from pelvis to goal
        double direction[2];
//        mju_sub(direction, goal_pos, pelvis_pos, 2);
//
//        // normalize direction vector
//        mju_normalize(direction, 2);
        double goal_angle = parameters_[3];
        // make goal angle from degrees to radians
        goal_angle = goal_angle * M_PI / 180.0;
        direction[0] = std::cos(goal_angle);
        direction[1] = std::sin(goal_angle);


        double orientation_error[2];


        mju_sub(orientation_error, direction, pelvis_x_axis, 2);

        mju_copy(&residual[counter], orientation_error, 2);
        mju_scl(&residual[counter], &residual[counter], standing * 10, 2);
        counter += 2;

//        printf("pelvis_x_axis: %f, %f, %f\n", pelvis_x_axis[0], pelvis_x_axis[1], pelvis_x_axis[2]);
//        printf("direction: %f, %f\n", direction[0], direction[1]);
//        printf("orientation_error: %f, %f\n", orientation_error[0], orientation_error[1]);
//        printf("-------------------\n");

//        // goal orientation
//        double goal_rotmat[9];
//        const double *goal_orientation = data->mocap_quat;
//        mju_quat2Mat(goal_rotmat, goal_orientation);

//        // system's orientation
//        double body_rotmat[9];
//        double *orientation = SensorByName(model, data, "pelvis_orientation");
//        mju_quat2Mat(body_rotmat, orientation);
//        double x_transformed[3];
//        x_transformed[0] = goal_rotmat[0];  // First element of the first row
//        x_transformed[1] = goal_rotmat[3];  // First element of the second row
//        x_transformed[2] = goal_rotmat[6];  // First element of the third row
//
//        mju_sub(&residual[counter], body_rotmat, goal_rotmat, 9);
//        mju_scl(&residual[counter], &residual[counter], standing, 9);
//        counter += 9;
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

// -------- Transition for humanoid_bench maze task -------- //
// ------------------------------------------------------------ //
    void H1_maze::TransitionLocked(mjModel *model, mjData *data) {
        double *goal_pos = model->key_mpos + 3 * (curr_goal_idx_ + 1);  // offset 1 is on purpose
        double *pelvis_pos = SensorByName(model, data, "pelvis_position");
        double dist = std::sqrt(std::pow(goal_pos[0] - pelvis_pos[0], 2) +
                                std::pow(goal_pos[1] - pelvis_pos[1], 2));


        // check if task is done
        if (dist < 0.3) {
            curr_goal_idx_ = std::min(curr_goal_idx_ + 1, 4);
        }
        mju_copy3(data->mocap_pos, model->key_mpos + 3 * (curr_goal_idx_ + 1)); // offset 1 is on purpose
    }

    void H1_maze::ResetLocked(const mjModel *model) {
        curr_goal_idx_ = 0;
    }

}  // namespace mjpc
