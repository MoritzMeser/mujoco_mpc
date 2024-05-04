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


        // ----- actuator velocity ----- //
        double vel_margin = parameters_[2];
        double actuator_velocity = 0.0;
        for (int i = 0; i < model->nu; i++) {

            x = data->actuator_velocity[i];
//            x = data->cvel[i];

            actuator_velocity += tolerance(x, {0, 0}, vel_margin, "quadratic", 0.0);
        }
        actuator_velocity /= model->nu;  // average over all controls
        actuator_velocity = (2 + actuator_velocity) / 3;

        reward *= actuator_velocity;
        // the moment seems to do nothing (most values are always zero anyway)

//        // ----- actuator moment ----- //
//        double moment_margin = parameters_[3];
//        double actuator_moment = 0.0;
//        for (int i = 0; i < model->nu; i++) {
//            x = data->actuator_moment[i];
//            printf("actuator_moment: %f\n", x);
//            actuator_moment += tolerance(x, {0, 0}, moment_margin, "gaussian", 0.1);
//        }
//        actuator_moment /= model->nu;  // average over all controls
//        actuator_moment = (2 + actuator_moment) / 3;
//
//        reward *= actuator_moment;

        // ----- foot height ----- //
        double right_foot_height = SensorByName(model, data, "right_foot_height")[2];
        double left_foot_height = SensorByName(model, data, "left_foot_height")[2];
        double right_foot_reward = tolerance(right_foot_height, {0.0, 0.2}, 0.1);
        double left_foot_reward = tolerance(left_foot_height, {0.0, 0.2}, 0.1);
        double foot_reward = (right_foot_reward + left_foot_reward) / 2;
        foot_reward = (4 + foot_reward) / 5;

        reward *= foot_reward;

//        // ----- foot distance ----- //
//        double left_foot_x = SensorByName(model, data, "left_foot_height")[0];
//        double right_foot_x = SensorByName(model, data, "right_foot_height")[0];
//        double left_foot_y = SensorByName(model, data, "left_foot_height")[1];
//        double right_foot_y = SensorByName(model, data, "right_foot_height")[1];
//        double foot_distance = std::sqrt(std::pow(left_foot_x - right_foot_x, 2) +
//                                         std::pow(left_foot_y - right_foot_y, 2));
//        foot_distance = tolerance(foot_distance, {0.2, 0.6}, 0.1);
//        foot_reward *= foot_distance;
//
//        foot_reward = (4 + foot_reward) / 5;

        // ----- reward computation ----- //
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

        // ----- residuals ----- //
        residual[0] = 1.0 - reward;
    }

// -------- Transition for H1 walk task --------
// for a more complex task this might be necessary (like walking to different targets)
// ---------------------------------------------
    void H1_walk::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
