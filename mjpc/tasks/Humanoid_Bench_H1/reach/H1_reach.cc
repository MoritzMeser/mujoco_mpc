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

#include "H1_reach.h"

#include <string>
#include <limits>
#include <cmath>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    std::string H1_reach::XmlPath() const {
        return GetModelPath("Humanoid_Bench_H1/reach/task.xml");
    }

    std::string H1_reach::Name() const { return "H1 Reach"; }

// ----------------- Residuals for Humanoid_Bench_H1 reach task ----------------

// -------------------------------------------------------------
    void H1_reach::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double *goal_pos = SensorByName(model, data, "goal_pos");
        double *left_hand_pos = SensorByName(model, data, "left_hand_position");
        double hand_dist = std::sqrt(std::pow(goal_pos[0] - left_hand_pos[0], 2) +
                                     std::pow(goal_pos[1] - left_hand_pos[1], 2) +
                                     std::pow(goal_pos[2] - left_hand_pos[2], 2));

        double healthy_reward = data->xmat[1 * 9 + 8];
        double motion_penalty = 0.0;
        for (int i = 0; i < model->nu; i++) {
            motion_penalty += data->qvel[i];
        }
        double reward_close = (hand_dist < 1) ? 5 : 0;
        double reward_success = (hand_dist < 0.05) ? 10 : 0;

        double reward = healthy_reward - 0.0001 * motion_penalty + reward_close + reward_success;
        residual[0] = 20 - reward;  // 20 is the maximum reward --> change this if the reward-computation changes
    }

// -------- Transition for Humanoid_Bench_H1 reach task -------- //
// ------------------------------------------------------------- //
    void H1_reach::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
