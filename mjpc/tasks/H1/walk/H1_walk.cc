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
#include <limits>
#include <cmath>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    std::string H1_walk::XmlPath() const {
        return GetModelPath("H1/walk/task.xml");
    }

    std::string H1_walk::Name() const { return "H1 Walk"; }

// ----------------- Residuals for H1 walk task ----------------

// -------------------------------------------------------------
    void H1_walk::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        // initialize reward
        double reward = 1.0;

        // get values from GUI
        double move_speed = parameters_[0];
        double stand_height = parameters_[1];

        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {stand_height, INFINITY}, stand_height / 4);

        reward *= standing;


        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9);

        reward *= upright;


        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
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
            double move = tolerance(com_velocity, {move_speed, INFINITY}, std::abs(move_speed), "linear", 0.0);
            move = (5 * move + 1) / 6;
            reward *= move;
        }

        // this is the reward as implemented in the original task in humanoid bench
        // https://humanoid-bench.github.io
        double const humanoid_bench_reward = reward;




        //  ----- --- --- --- --- --- ----- //
        // ----- modification of reward ----- //
        //  ----- --- --- --- --- --- ----- //



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
        double total_reward =
                tradeoff * humanoid_bench_reward + (1 - tradeoff) * humanoid_bench_reward * vel_reward * foot_reward;
        residual[0] = 1.0 - total_reward;
    }

// -------- Transition for H1 walk task --------
// for a more complex task this might be necessary (like walking to different targets)
// ---------------------------------------------
    void H1_walk::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
