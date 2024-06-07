//
// Created by Moritz Meser on 15.05.24.
//

#include "H1_stand.h"

#include <string>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/basic_locomotion/walk_reward.h"
#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"


namespace mjpc {
// ----------------- Residuals for humanoid_bench stand task ---------------- //
// ----------------------------------------------------------------------------- //
    void H1_stand::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double const walk_speed = 0.0;
        double const stand_height = 1.65;

        int residual_counter = 0;
        // ----- original humanoid bench reward ----- //
        double reward = walk_reward(model, data, walk_speed, stand_height);
        residual[residual_counter++] = std::exp(-reward);  // 1.

        // ----- individual reward terms ----- //

        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {stand_height, INFINITY}, stand_height / 4);

        residual[residual_counter++] = 1.0 - standing; // 2.


        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9);

        residual[residual_counter++] = 1.0 - upright; // 3.


//        // ----- small control ----- //
//        double small_control = 0.0;
//        for (int i = 0; i < model->nu; i++) {
//            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
//        }
//        small_control /= model->nu;  // average over all controls
//
//        residual[residual_counter++] = 1.0 - small_control; // 4.
//
////        // ----- small moment ----- //
////        double small_moment = 0.0;
////        for (int i = 0; i < model->nu; i++) {
////            small_control += tolerance(data->actuator_moment[i] , {0.0, 0.0}, 10.0, "quadratic", 0.0);
////        }
////        small_moment /= model->nu;  // average over all controls
////
////        residual[residual_counter++] = 1.0 - small_moment;
//
//        // ----- small force ----- //
//        double average_force = 0.0;
//        for (int i = 0; i < model->nu; i++) {
//            average_force += std::abs(data->actuator_force[i]);
//        }
//        average_force /= model->nu;
//        residual[residual_counter++] = average_force; // 5.
//
//
////        // ----- small acceleration ----- //
////        double small_acceleration = 0.0;
////        for (int i = 0; i < model->nv; i++) {
////            small_acceleration += tolerance(data->qacc[i] , {0.0, 0.0}, 10.0, "quadratic", 0.0);
////        }
////        small_acceleration /= model->nv;  // average over all controls
////
////        residual[residual_counter++] = 1.0 - small_acceleration;
//
//        // ----- average acceleration ----- //
//        double average_acceleration = 0.0;
//        for (int i = 0; i < model->nv; i++) {
//            average_acceleration += std::abs(data->qacc[i]);
//        }
//        average_acceleration /= model->nv;
//        residual[residual_counter++] = average_acceleration; // 6.

        // ----- move speed ----- //
        double move_reward = 1.0;
        if (walk_speed == 0.0) {
            double horizontal_velocity_x = SensorByName(model, data, "center_of_mass_velocity")[0];
            double horizontal_velocity_y = SensorByName(model, data, "center_of_mass_velocity")[1];
            double dont_move = (tolerance(horizontal_velocity_x, {0.0, 0.0}, 2) +
                                tolerance(horizontal_velocity_y, {0.0, 0.0}, 2)) / 2;
            move_reward = dont_move;
        } else {
            double com_velocity = SensorByName(model, data, "center_of_mass_velocity")[0];
            double move = tolerance(com_velocity, {walk_speed, INFINITY}, std::abs(walk_speed), "linear", 0.0);
            move = (5 * move + 1) / 6;
            move_reward = move;
        }

        residual[residual_counter++] = 1.0 - move_reward; // 7.

        // ----- joint velocity ----- //
        mju_copy(residual + residual_counter, data->qvel + 6, model->nv - 6); // 8.
        residual_counter += model->nv - 6;

//        // ----- action ----- //
//        mju_copy(&residual[residual_counter], data->ctrl, model->nu);
//        residual_counter += model->nu;

        // ----- effort ----- //
        mju_scl(residual + residual_counter, data->actuator_force, 2e-2, model->nu);
        residual_counter += model->nu;


        // ----- posture ----- //
        mju_copy(&residual[residual_counter], data->qpos + 7, model->nq - 7);
        residual_counter += model->nq - 7;

        // ----- COM xy velocity should be 0 ----- //
        double* com_velocity = SensorByName(model, data, "center_of_mass_velocity");
        mju_copy(&residual[residual_counter], com_velocity, 2);
        residual_counter += 2;

        // sensor dim sanity check
        // TODO: use this pattern everywhere and make this a utility function
        int user_sensor_dim = 0;
        for (int i = 0; i < model->nsensor; i++) {
            if (model->sensor_type[i] == mjSENS_USER) {
                user_sensor_dim += model->sensor_dim[i];
            }
        }
        if (user_sensor_dim != residual_counter) {
            mju_error_i(
                    "mismatch between total user-sensor dimension "
                    "and actual length of residual %d",
                    residual_counter);
        }
    }
}  // namespace mjpc
