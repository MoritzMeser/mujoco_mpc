//
// Created by Moritz Meser on 15.05.24.
//

#include "H1_walk.h"

#include <string>
#include <limits>
#include <cmath>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/utility/dm_control_utils_rewards.h"
#include "mjpc/tasks/Humanoid_Bench_H1/basic_locomotion/compute_basic_reward.h"


namespace mjpc {
    std::string H1_walk::XmlPath() const {
        return GetModelPath("Humanoid_Bench_H1/basic_locomotion/walk/task.xml");
    }

    std::string H1_walk::Name() const { return "H1 Walk"; }

// ----------------- Residuals for Humanoid_Bench_H1 walk task ----------------

// -------------------------------------------------------------
    void H1_walk::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double const walk_speed = 1.0;
        double const stand_height = 1.65;

        double reward = compute_basic_reward(model, data, walk_speed, stand_height);
        residual[0] = 1 - reward;
    }
}  // namespace mjpc
