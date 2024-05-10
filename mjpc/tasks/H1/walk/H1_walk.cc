// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "H1_walk.h"

#include <string>
# include <limits>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/H1/utility/reward_computation.h"

namespace mjpc {
    std::string H1_walk::XmlPath() const {
        return GetModelPath("H1/walk/task.xml");
    }

    std::string H1_walk::Name() const { return "H1 Walk"; }

// ----------------- Residuals for H1 walk task ----------------

// -------------------------------------------------------------
    void H1_walk::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                       double *residual) const {
        // initialize reward
        double reward = 1.0;

        // get values from GUI
        double stand_height = parameters_[1];
        double move_speed = parameters_[0];

        // ----- standing ----- //
        double infinity = std::numeric_limits<double>::infinity();
        std::pair<double, double> bounds = {stand_height, infinity};

        double x = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(x, bounds, stand_height / 4);

        reward *= standing;


        // ----- torso upright ----- //
        bounds = {0.9, infinity};
        x = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(x, bounds, 1.9);

        reward *= upright;


        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            double margin = 10;
            double value_at_margin = 0.0;
            x = data->ctrl[i];
            small_control += tolerance(x, {0.0, 0.0}, margin, "quadratic", value_at_margin);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        reward *= small_control;

        // ----- move speed ----- //
        if (move_speed == 0.0) {
            double horizontal_velocity_x = SensorByName(model, data, "center_of_mass_velocity")[0];
            double horizontal_velocity_y = SensorByName(model, data, "center_of_mass_velocity")[1];
            double dont_move = (tolerance(horizontal_velocity_x, {0.0, 0.0}, 2) +
                                tolerance(horizontal_velocity_y, {0.0, 0.0}, 2)) / 2;
            reward *= dont_move;
        } else {
            double com_velocity = SensorByName(model, data, "center_of_mass_velocity")[0];
            double move = tolerance(com_velocity, {move_speed, infinity}, std::abs(move_speed), "linear", 0.0);
            move = (5 * move + 1) / 6;
            reward *= move;
        }

        //  ----- --- --- --- --- --- ----- //
        // ----- modification of reward ----- //
        //  ----- --- --- --- --- --- ----- //


//        // ----- actuator velocity ----- //
//        double vel_margin = parameters_[2];
//        double vel_bound = parameters_[3];
////        double actuator_velocity = 0.0;
//        for (int i = 0; i < model->nu; i++) {
////            actuator_velocity += tolerance(data->actuator_velocity[i], {-vel_bound, +vel_bound}, vel_margin, "quadratic", 0.0);
//            reward *= (3 +
//                       tolerance(data->actuator_velocity[i], {-vel_bound, +vel_bound}, vel_margin, "linear", 0.2)) /
//                      4;
//
//        }
//        actuator_velocity /= model->nu;  // average over all controls
//        actuator_velocity = (2 + actuator_velocity) / 3;

//        reward *= actuator_velocity;

//        // ----- foot height ----- //
//        double right_foot_height = SensorByName(model, data, "right_foot_height")[2];
//        double left_foot_height = SensorByName(model, data, "left_foot_height")[2];
//        double right_foot_reward = tolerance(right_foot_height, {0.0, 0.1}, 0.1);
//        double left_foot_reward = tolerance(left_foot_height, {0.0, 0.1}, 0.1);
//        double foot_reward = (right_foot_reward + left_foot_reward) / 2;
////        foot_reward = (4 + foot_reward) / 5;
//
//        reward *= foot_reward;
//
////        // ----- foot position ----- //
//////        double *pelvis_x_orientation = SensorByName(model, data, "pelvis_x_orientation");
//////        double *pelvis_y_orientation = SensorByName(model, data, "pelvis_y_orientation");
//////        double *pelvis_z_orientation = SensorByName(model, data, "pelvis_z_orientation");
////
////        // Get the x, y, and z axes of the pelvis
////        mjtNum *pelvis_x_axis = SensorByName(model, data, "pelvis_x_orientation");
////        mjtNum *pelvis_y_axis = SensorByName(model, data, "pelvis_y_orientation");
////        mjtNum *pelvis_z_axis = SensorByName(model, data, "pelvis_z_orientation");
////
////// Form the rotation matrix of the pelvis
////        mjtNum pelvis_rot[9];
////        for (int i = 0; i < 3; i++) {
////            pelvis_rot[i] = pelvis_x_axis[i];
////            pelvis_rot[i + 3] = pelvis_y_axis[i];
////            pelvis_rot[i + 6] = pelvis_z_axis[i];
////        }
////
////        double *left_foot_pos_global = SensorByName(model, data, "left_foot_height");
////        double *right_foot_pos_global = SensorByName(model, data, "right_foot_height");
////        double *pelvis_pos_global = SensorByName(model, data, "pelvis_position");
////
////
////// Compute the position of the left foot in the coordinate frame of the pelvis
////        mjtNum left_foot_pos_pelvis_global[3];
////        for (int i = 0; i < 3; i++) {
////            left_foot_pos_pelvis_global[i] = left_foot_pos_global[i] - pelvis_pos_global[i];
////        }
////        mjtNum left_foot_pos_pelvis[3];
////        rotateVector(left_foot_pos_pelvis, pelvis_rot, left_foot_pos_pelvis_global);
////
////// Compute the position of the right foot in the coordinate frame of the pelvis
////        mjtNum right_foot_pos_pelvis_global[3];
////        for (int i = 0; i < 3; i++) {
////            right_foot_pos_pelvis_global[i] = right_foot_pos_global[i] - pelvis_pos_global[i];
////        }
////        mjtNum right_foot_pos_pelvis[3];
////        rotateVector(right_foot_pos_pelvis, pelvis_rot, right_foot_pos_pelvis_global);
////
////        // ----- foot distance ----- //
////        double foot_position_reward = 1.0;
////        foot_position_reward *= tolerance(left_foot_pos_pelvis[0], {-0.3, 0.3}, 0.2, "linear", 0.0);  // x position of each foot should be around zero
////        foot_position_reward *= tolerance(left_foot_pos_pelvis[1], {-1.0, 1.0}, 0.2, "linear", 0.0);  // y position of left foot should be negative TODO: check this
////        foot_position_reward *= tolerance(right_foot_pos_pelvis[0], {-0.3, 0.3}, 0.2, "linear", 0.0);  // x position of each foot should be around zero
//////        foot_position_reward *= tolerance(right_foot_pos_pelvis[1], {0.2, 0.6}, 0.2, "linear", 0.0);  // y position of right foot should be positive TODO: check this
////
////        reward *= foot_position_reward;
//
//        // ----- hand height ----- //
//        double right_hand_height = SensorByName(model, data, "right_hand_position")[2];
//        double left_hand_height = SensorByName(model, data, "left_hand_position")[2];
//        double right_hand_reward = tolerance(right_hand_height, {0.0, 1.8}, 0.1);  // not much above the head height
//        double left_hand_reward = tolerance(left_hand_height, {0.0, 1.8}, 0.1);  // not much above the head height
//        double hand_reward = (right_hand_reward + left_hand_reward) / 2;
//        reward *= hand_reward;
//
//
//        // ----- hand velocity ----- //
//        double *right_hand_velocity = SensorByName(model, data, "right_hand_velocity");
//        double *left_hand_velocity = SensorByName(model, data, "left_hand_velocity");
//        double right_hand_speed = std::sqrt(right_hand_velocity[0] * right_hand_velocity[0] +
//                                            right_hand_velocity[1] * right_hand_velocity[1] +
//                                            right_hand_velocity[2] * right_hand_velocity[2]);
//        double left_hand_speed = std::sqrt(left_hand_velocity[0] * left_hand_velocity[0] +
//                                           left_hand_velocity[1] * left_hand_velocity[1] +
//                                           left_hand_velocity[2] * left_hand_velocity[2]);
//        reward *= tolerance(right_hand_speed, {0.0, 0.6}, 0.05, "linear", 0.0);
//        reward *= tolerance(left_hand_speed, {0.0, 0.6}, 0.05, "linear", 0.0);


        // ----- actuator velocity ----- //
        double vel_margin = parameters_[2];
        double vel_bound = parameters_[3];
        double vel_reward = 1.0;
        for (int i = 0; i < model->nu; i++) {
            vel_reward *= tolerance(data->actuator_velocity[i], {-vel_bound, +vel_bound}, vel_margin, "quadratic", 0.0);
        }
        //----- foot height ----- //
        double right_foot_height = SensorByName(model, data, "right_foot_height")[2];
        double left_foot_height = SensorByName(model, data, "left_foot_height")[2];
        double right_foot_reward = tolerance(right_foot_height, {0.0, 0.1}, 0.1);
        double left_foot_reward = tolerance(left_foot_height, {0.0, 0.1}, 0.1);
        double foot_reward = (right_foot_reward + left_foot_reward) / 2;

        // ----- residuals ----- //
        // idea: use a sum of two terms for the reward. First, the normal reward, second, the normal reward times the vel_reward.
        // This should give a positive reward even if a slow velocity cannot be achieved
        // at the same time, just no movement on the floor gives no positive reward
        double tradeoff = parameters_[4];
        double total_reward = tradeoff * reward + (1 - tradeoff) * reward * vel_reward * foot_reward;
        residual[0] = 1.0 - total_reward;
    }

// -------- Transition for H1 walk task --------
// for a more complex task this might be necessary (like walking to different targets)
// ---------------------------------------------
    void H1_walk::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
