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
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    std::string H1_package::XmlPath() const {
        return GetModelPath("Humanoid_Bench_H1/package/task.xml");
    }

    std::string H1_package::Name() const { return "H1 Package"; }

// ----------------- Residuals for Humanoid_Bench_H1 package task ----------------

// -------------------------------------------------------------
    void H1_package::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                          double *residual) const {
        // get values from GUI
        double stand_height = parameters_[1];

        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {stand_height, INFINITY}, stand_height / 4);


        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9);

        double stand_reward = standing * upright;


        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        // ----- rewards specific to the package task ----- //

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
        double const humanoid_bench_reward = (
                stand_reward * small_control
                - 3 * dist_package_destination
                - (dist_hand_package_left + dist_hand_package_right) * 0.1
                + package_height
                + reward_success * 1000
        );


        // ----- residuals ----- //

        // because the reward in this task is not bound to [0, 1] we need to use the exponential function
        residual[0] = std::exp(-humanoid_bench_reward);
    }

// -------- Transition for Humanoid_Bench_H1 package task --------
// for a more complex task this might be necessary (like packageing to different targets)
// ---------------------------------------------
    void H1_package::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
