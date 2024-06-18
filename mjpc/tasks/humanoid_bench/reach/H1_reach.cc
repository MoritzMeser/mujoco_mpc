#include "H1_reach.h"

#include <string>
#include <limits>
#include <cmath>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include <array>
#include <random>

namespace mjpc {
// ----------------- Residuals for humanoid_bench reach task ---------------- //
// ----------------------------------------------------------------------------- //
    void H1_reach::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double const height_goal = parameters_[0];
        double const walk_speed = parameters_[1];


        //
        //
        //

        // print initial position of pelvis
//        double *pelvis_position = SensorByName(model, data, "pelvis_position");
//        printf("pelvis_position: %f %f %f\n", pelvis_position[0], pelvis_position[1], pelvis_position[2]);
//
//        // print initial quaternion of pelvis
//        double *pelvis_orientation = SensorByName(model, data, "pelvis_orientation");
//        printf("pelvis_orientation: %f %f %f %f\n", pelvis_orientation[0], pelvis_orientation[1], pelvis_orientation[2], pelvis_orientation[3]);
//        pelvis_position: -0.000163 0.000612 0.992704
//        pelvis_orientation: 0.483195 0.485925 -0.512847 0.517095
        //qpos="0 0 0.98 1 0 0 0 0
        //
        //
        //

//        data->qpos[0] = 0;
//        data->qpos[1] = 0;
//        data->qpos[2] = 0.98;
//        data->qpos[3] = 1;
//        data->qpos[4] = 0;
//        data->qpos[5] = 0;
//        data->qpos[6] = 0;
//
        int counter = 0;

        // ----- Height: head feet vertical error ----- //

        // feet sensor positions
        double *foot_right_pos = SensorByName(model, data, "foot_right_pos");
        double *foot_left_pos = SensorByName(model, data, "foot_left_pos");

        double *head_position = SensorByName(model, data, "head_position");
        double head_feet_error =
                head_position[2] - 0.5 * (foot_right_pos[2] + foot_left_pos[2]);
        residual[counter++] = head_feet_error - height_goal;

        // ----- Balance: CoM-feet xy error ----- //

//        // capture point
//        double *com_position = SensorByName(model, data, "torso_subtreecom");
        double *com_velocity = SensorByName(model, data, "torso_subtreelinvel");
//        double kFallTime = 0.2;
//        double capture_point[3] = {com_position[0], com_position[1], com_position[2]};
//        mju_addToScl3(capture_point, com_velocity, kFallTime);
//
//        // average feet xy position
//        double fxy_avg[2] = {0.0};
//        mju_addTo(fxy_avg, foot_right_pos, 2);
//        mju_addTo(fxy_avg, foot_left_pos, 2);
//        mju_scl(fxy_avg, fxy_avg, 0.5, 2);
//
//        mju_subFrom(fxy_avg, capture_point, 2);
//        double com_feet_distance = mju_norm(fxy_avg, 2);
//        residual[counter++] = com_feet_distance;

        // ----- COM xy velocity should be 0 ----- //
        if (std::abs(walk_speed) < 1e-3) {
            mju_copy(&residual[counter], com_velocity, 2);
        } else {
            mju_copy(&residual[counter], com_velocity, 2);
            residual[counter] = 0.0;
        }
        counter += 2;

        // ----- joint velocity ----- //
        mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
        counter += model->nv - 6;
        double sum_qvel = 0;
        for (int i = 0; i < model->nv - 6; i++) {
            sum_qvel += std::abs(data->qvel[6 + i]);
        }
        double q_vel_low = tolerance(sum_qvel, {-10, +10}, 5.0, "quadratic", 0.0);

//        // ----- action ----- //
//        mju_copy(&residual[counter], data->ctrl, model->nu);
//        counter += model->nu;

        // ----- reach task ----- //
//        double *goal_pos = SensorByName(model, data, "goal_pos");
//        double *left_hand_pos = SensorByName(model, data, "left_hand_position");
//        double hand_dist = std::sqrt(std::pow(goal_pos[0] - left_hand_pos[0], 2) +
//                                     std::pow(goal_pos[1] - left_hand_pos[1], 2) +
//                                     std::pow(goal_pos[2] - left_hand_pos[2], 2));
//
//
//

//        // ----- torso height ----- //
        double torso_height = SensorByName(model, data, "torso_position")[2];
//        residual[counter++] = torso_height - parameters_[0];

//        // ----- pelvis / feet ----- //
//        double *foot_right = SensorByName(model, data, "foot_right");
//        double *foot_left = SensorByName(model, data, "foot_left");
//        double pelvis_height = SensorByName(model, data, "pelvis_position")[2];
//        residual[counter++] =
//                0.5 * (foot_left[2] + foot_right[2]) - pelvis_height - 0.2;

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

        // is standing
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

        // ----- keep initial position -----//
        residual[counter + 0] = data->qpos[0];
        residual[counter + 1] = data->qpos[1];
        residual[counter + 2] = 1.0 - data->qpos[2];
        residual[counter + 3] = 1.0 - data->qpos[3];
        residual[counter + 4] = data->qpos[4];
        residual[counter + 5] = data->qpos[5];
        residual[counter + 6] = data->qpos[6];

        counter += 7;


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

        // com vel
        double *waist_lower_subcomvel =
                SensorByName(model, data, "waist_lower_subcomvel");
        double *torso_velocity = SensorByName(model, data, "torso_velocity");
        double com_vel[2];
        mju_add(com_vel, waist_lower_subcomvel, torso_velocity, 2);
        mju_scl(com_vel, com_vel, 0.5, 2);

        // Walk forward
        residual[counter++] =
                standing * (mju_dot(com_vel, forward, 2) - walk_speed);

        // ----- move feet ----- //
        double *foot_right_vel = SensorByName(model, data, "foot_right_vel");
        double *foot_left_vel = SensorByName(model, data, "foot_left_vel");
        double move_feet[2];
        mju_copy(move_feet, com_vel, 2);
        mju_addToScl(move_feet, foot_right_vel, -0.5, 2);
        mju_addToScl(move_feet, foot_left_vel, -0.5, 2);

        mju_copy(&residual[counter], move_feet, 2);
        mju_scl(&residual[counter], &residual[counter], standing, 2);
        counter += 2;

        // ----- control ----- //
        mju_sub(&residual[counter], data->ctrl, model->key_qpos + 7, model->nu); // because of pos control
        counter += model->nu;

//         ----- reach task ----- //
        double *goal_pos = SensorByName(model, data, "goal_pos");
        double *left_hand_pos = SensorByName(model, data, "left_hand_pos");
        double *right_hand_pos = SensorByName(model, data, "right_hand_pos");
//        printf(task_->use_left_hand_ ? "left hand in residual\n" : "right hand in residual\n");


        if (task_->use_left_hand_) {
            mju_sub3(&residual[counter], goal_pos, left_hand_pos);
        } else {
            mju_sub3(&residual[counter], goal_pos, right_hand_pos);
        }

        mju_scl(&residual[counter], &residual[counter], standing * q_vel_low, 3);
        counter += 3;

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


// -------- Transition for humanoid_bench reach task -------- //
// ------------------------------------------------------------- //
    void H1_reach::TransitionLocked(mjModel *model, mjData *data) {

        double *hand_pos;
        if (use_left_hand_) {
            hand_pos = SensorByName(model, data, "left_hand_pos");
        } else {
            hand_pos = SensorByName(model, data, "right_hand_pos");
        }
        double hand_dist = std::sqrt(std::pow(target_position_[0] - hand_pos[0], 2) +
                                     std::pow(target_position_[1] - hand_pos[1], 2) +
                                     std::pow(target_position_[2] - hand_pos[2], 2));
        // check if task is done
        if ((hand_dist < 0.05 || hand_dist > 100) && data->time > 0.1) {
            // generate new random target
            std::array<double, 3> target_low = {0, -0.20, -0.30};
            std::array<double, 3> target_high = {0.20, 0.20, 0.30};
            std::random_device rd;
            std::mt19937 gen(rd());
            std::array<double, 3> new_target = {0, 0, 0};
            for (int i = 0; i < 3; ++i) {
                std::uniform_real_distribution<> dis(target_low[i], target_high[i]);
                new_target[i] = dis(gen);
            }

            // add offset to the left hand site or right hand site
            double left_hand_offset[3] = {0.3, -0.2, -0.1};
            double right_hand_offset[3] = {0.3, 0.2, -0.1};
            if (use_left_hand_) {
                mju_add3(new_target.data(), new_target.data(), left_hand_offset);
            } else {
                mju_add3(new_target.data(), new_target.data(), right_hand_offset);
            }


            // move new target to front of robot
            double *pelvis_position = SensorByName(model, data, "torso_position");
            double *pelvis_orientation = SensorByName(model, data, "torso_orientation");

            // Compute the relative target position in the local coordinate system of the pelvis
            double relative_target_pos[3] = {new_target[0], new_target[1], new_target[2]};

            // Convert the pelvis's orientation quaternion to a rotation matrix
            double pelvis_rotmat[9];
            mju_quat2Mat(pelvis_rotmat, pelvis_orientation);

            // Multiply the relative target position by the rotation matrix to rotate it according to the pelvis's orientation
            double rotated_relative_target_pos[3];
            mju_mulMatVec(rotated_relative_target_pos, pelvis_rotmat, relative_target_pos, 3, 3);

            // Add the rotated relative target position to the pelvis's position to get the target position in the global coordinate system
            double global_target_pos[3];
            mju_add(global_target_pos, pelvis_position, rotated_relative_target_pos, 3);

            // Copy the global target position to mocap_pos
            target_position_[0] = global_target_pos[0];
            target_position_[1] = global_target_pos[1];
            target_position_[2] = global_target_pos[2];

            printf("new target %f %f %f\n", target_position_[0], target_position_[1], target_position_[2]);
            use_left_hand_ = !use_left_hand_;

        }
        mju_copy3(data->mocap_pos, target_position_.data());
        mju_copy3(data->mocap_pos + 3, hand_pos);
    }


}  // namespace mjpc