//
// Created by Moritz Meser on 15.05.24.
//

#include "H1_stand.h"

#include <string>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/Humanoid_Bench_H1/basic_locomotion/walk_reward.h"


namespace mjpc {
    std::string H1_stand::XmlPath() const {
        return GetModelPath("Humanoid_Bench_H1/basic_locomotion/stand/task.xml");
    }

    std::string H1_stand::Name() const { return "H1 Stand"; }

// ----------------- Residuals for Humanoid_Bench_H1 stand task ----------------

// -------------------------------------------------------------
    void H1_stand::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double const walk_speed = 0.0;
        double const stand_height = 1.65;

        double reward = walk_reward(model, data, walk_speed, stand_height);
        residual[0] = 1 - reward;

    }
}  // namespace mjpc
