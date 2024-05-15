//
// Created by Moritz Meser on 15.05.24.
//

#include "H1_stairs.h"

#include <string>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/Humanoid_Bench_H1/basic_locomotion/climbing_upwards_reward.h"


namespace mjpc {
    std::string H1_stairs::XmlPath() const {
        return GetModelPath("Humanoid_Bench_H1/basic_locomotion/stairs/task.xml");
    }

    std::string H1_stairs::Name() const { return "H1 Stairs"; }

// ----------------- Residuals for Humanoid_Bench_H1 walk task ----------------

// -------------------------------------------------------------
    void H1_stairs::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double const walk_speed = 1.0;

        double reward = climbing_upwards_reward(model, data, walk_speed);
        residual[0] = 1 - reward;
    }
}  // namespace mjpc
