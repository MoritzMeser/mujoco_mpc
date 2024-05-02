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

#include "package.h"

#include <string>
# include <limits>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/H1/utility/reward_computation.h"

namespace mjpc {
    std::string H1_package::XmlPath() const {
        return GetModelPath("H1/package/task_package.xml");
    }

    std::string H1_package::Name() const { return "H1 Package"; }

// ----------------- Residuals for H1 package task ----------------

// -------------------------------------------------------------
    void H1_package::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                          double *residual) const {
        // get values from GUI
        double stand_height = parameters_[1];
//        double move_speed = parameters_[0];

        // ----- standing ----- //
        double infinity = std::numeric_limits<double>::infinity();
        std::pair<double, double> bounds = {stand_height, infinity};

        double x = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(x, bounds, stand_height / 4);


        // ----- torso upright ----- //
        bounds = {0.9, infinity};
        x = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(x, bounds, 1.9);

        // ----- stand_reward ----- //
        double stand_reward = standing * upright;


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


        // ----- actuator velocity ----- //
        double margin = parameters_[2];
        double vel_bounds = parameters_[3];
        double actuator_velocity = 0.0;
        for (int i = 0; i < model->nu; i++) {
            double value_at_margin = 0.0;
            x = data->actuator_velocity[i];
            actuator_velocity += tolerance(x, {-vel_bounds, value_at_margin}, margin, "quadratic", value_at_margin);
        }
        actuator_velocity /= model->nu;  // average over all controls
        actuator_velocity = (2 + actuator_velocity) / 3;

        // ----- foot height ----- //
        double right_foot_height = SensorByName(model, data, "right_foot_height")[2];
        double left_foot_height = SensorByName(model, data, "left_foot_height")[2];
        double right_foot_reward = tolerance(right_foot_height, {0.0, 0.2}, 0.1);
        double left_foot_reward = tolerance(left_foot_height, {0.0, 0.2}, 0.1);
        double foot_reward = (right_foot_reward + left_foot_reward) / 2;
        foot_reward = (4 + foot_reward) / 5;


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

#include <cmath>
#include <algorithm>

// Assuming package_location, package_destination, right_hand_location, and left_hand_location are std::array<double, 3>
        double *package_location = SensorByName(model, data, "package_location");
        double *package_destination = SensorByName(model, data, "package_destination");
        double *left_hand_location = SensorByName(model, data, "left_hand_position");
        double *right_hand_location = SensorByName(model, data, "right_hand_position");

        double dist_package_destination = std::hypot(
                std::hypot(package_location[0] - package_destination[0], package_location[1] - package_destination[1]),
                package_location[2] - package_destination[2]);

        double dist_hand_package_right = std::hypot(
                std::hypot(right_hand_location[0] - package_location[0], right_hand_location[1] - package_location[1]),
                right_hand_location[2] - package_location[2]);

        double dist_hand_package_left = std::hypot(
                std::hypot(left_hand_location[0] - package_location[0], left_hand_location[1] - package_location[1]),
                left_hand_location[2] - package_location[2]);

        double package_height = std::min(package_location[2], 1.0);

        bool reward_success = dist_package_destination < 0.1;

        // ----- reward computation ----- //
        double reward = (
                stand_reward * small_control
                - 3 * dist_package_destination
                - (dist_hand_package_left + dist_hand_package_right) * 0.1
                + package_height
                + reward_success * 1000
        );


        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for H1 package task --------
// for a more complex task this might be necessary (like packageing to different targets)
// ---------------------------------------------
    void H1_package::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
