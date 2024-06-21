//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_basketball.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench basketball task ---------------- //
// ---------------------------------------------------------------------------------- //
    void H1_basketball::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                             double *residual) const {
        // ----- set parameters ----- //
        double const standHeight = 1.65;

        // Initialize reward
        double reward = 0.0;

        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_position")[2];
        double hb_standing = tolerance(head_height, {standHeight, INFINITY}, standHeight / 4);

        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_up")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9, "linear", 0.0);

        double stand_reward = hb_standing * upright;

        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        // ----- hand proximity reward ----- //
        double *basketball_pos = SensorByName(model, data, "basketball");

        // Get the position vectors for left hand and right hand
        double *left_hand_pos = SensorByName(model, data, "left_hand_pos");
        double *right_hand_pos = SensorByName(model, data, "right_hand_pos");

        // Compute the Euclidean distance from each hand to the basketball
        double left_hand_distance = std::sqrt(
                std::pow(left_hand_pos[0] - basketball_pos[0], 2) +
                std::pow(left_hand_pos[1] - basketball_pos[1], 2) +
                std::pow(left_hand_pos[2] - basketball_pos[2], 2));
        double right_hand_distance = std::sqrt(
                std::pow(right_hand_pos[0] - basketball_pos[0], 2) +
                std::pow(right_hand_pos[1] - basketball_pos[1], 2) +
                std::pow(right_hand_pos[2] - basketball_pos[2], 2));
        double reward_hand_proximity = tolerance(std::max(left_hand_distance, right_hand_distance), {0, 0.2}, 1);

        // ----- ball success reward ----- //
        double ball_hoop_distance = std::sqrt(
                std::pow(basketball_pos[0] - SensorByName(model, data, "hoop_center")[0], 2) +
                std::pow(basketball_pos[1] - SensorByName(model, data, "hoop_center")[1], 2) +
                std::pow(basketball_pos[2] - SensorByName(model, data, "hoop_center")[2], 2));
        double reward_ball_success = tolerance(ball_hoop_distance, {0.0, 0.0}, 7, "linear");

        // ----- stage ----- //
        static std::string stage = "catch";
        if (stage == "catch") {
            int const ball_collision_id = mj_name2id(model, mjOBJ_GEOM, "basketball_collision");
            for (int i = 0; i < data->ncon; i++) {
                if (data->contact[i].geom1 == ball_collision_id || data->contact[i].geom2 == ball_collision_id) {
                    stage = "throw";
                    break;
                }
            }
        }

        if (stage == "throw") {
            reward = 0.15 * (stand_reward * small_control) + 0.05 * reward_hand_proximity +
                     0.8 * reward_ball_success;
        } else if (stage == "catch") {
            reward = 0.5 * (stand_reward * small_control) + 0.5 * reward_hand_proximity;
        }

        if (ball_hoop_distance < 0.05) {
            reward += 1000;
        }

        // ----- residuals ----- //
        int counter = 0;
        residual[counter++] = std::exp(-reward);


        double const height_goal = parameters_[0];



        // ----- Height: head feet vertical error ----- //

        // feet sensor positions
        double *foot_right_pos = SensorByName(model, data, "foot_right_pos");
        double *foot_left_pos = SensorByName(model, data, "foot_left_pos");

        double *head_position = SensorByName(model, data, "head_position");
        double head_feet_error =
                head_position[2] - 0.5 * (foot_right_pos[2] + foot_left_pos[2]);
        residual[counter++] = head_feet_error - height_goal;

        // ----- Balance: CoM-feet xy error ----- //

        // capture point
        double *com_velocity = SensorByName(model, data, "torso_subtreelinvel");


        // ----- COM xy velocity should be 0 ----- //
        mju_copy(&residual[counter], com_velocity, 2);
        counter += 2;


        // ----- joint velocity ----- //
        mju_copy(residual + counter, data->qvel + 6, model->nu);
        counter += model->nu;

//        double sum_qvel = 0;
//        for (int i = 0; i < model->nv - 6; i++) {
//            sum_qvel += std::abs(data->qvel[6 + i]);
//        }
//        double q_vel_low = tolerance(sum_qvel, {-10, +10}, 5.0, "quadratic", 0.0);



        // ----- torso height ----- //
        double torso_height = SensorByName(model, data, "torso_position")[2];

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
//        residual[counter + 0] = 0.0;
//        residual[counter + 1] = 0.0;
//        residual[counter + 2] = 0.0;
        residual[counter + 3] = 1.0 - data->qpos[3];
        residual[counter + 4] = data->qpos[4];
        residual[counter + 5] = data->qpos[5];
        residual[counter + 6] = data->qpos[6];

        counter += 7;

// ----- posture ----- //
        mju_sub(&residual[counter], data->qpos + 7, model->key_qpos + 7, model->nu);
        counter += model->nu;

        // com vel
        double *waist_lower_subcomvel =
                SensorByName(model, data, "waist_lower_subcomvel");
        double *torso_velocity = SensorByName(model, data, "torso_velocity");
        double com_vel[2];
        mju_add(com_vel, waist_lower_subcomvel, torso_velocity, 2);
        mju_scl(com_vel, com_vel, 0.5, 2);

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

        double* basketball = SensorByName(model, data, "basketball");
        double* hoop_center = SensorByName(model, data, "hoop_center");

        // ----- goal dist ------ //
        mju_sub3(&residual[counter], basketball, hoop_center);
        counter += 3;

        // ----- left hand dist ------ //
        double* left_hand = SensorByName(model, data, "left_hand_pos");
        mju_sub3(&residual[counter], left_hand, basketball);
        counter += 3;

        // ----- right hand dist ------ //
        double* right_hand = SensorByName(model, data, "right_hand_pos");
        mju_sub3(&residual[counter], right_hand, basketball);
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

// -------- Transition for humanoid_bench basketball task -------- //
// ------------------------------------------------------------ //
    void H1_basketball::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
