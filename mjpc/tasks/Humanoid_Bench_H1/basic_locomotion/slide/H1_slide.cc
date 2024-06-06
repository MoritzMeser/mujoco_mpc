//
// Created by Moritz Meser on 15.05.24.
//

#include "H1_slide.h"

#include <string>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/Humanoid_Bench_H1/basic_locomotion/climbing_upwards_reward.h"


namespace mjpc {
// ----------------- Residuals for Humanoid_Bench_H1 walk task ---------------- //
// ---------------------------------------------------------------------------- //
    void H1_slide::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double const walk_speed = 1.0;

        double reward = climbing_upwards_reward(model, data, walk_speed);
        residual[0] = std::exp(-reward);
    }
}  // namespace mjpc
