//
// Created by Moritz Meser on 20.05.24.
//

#include "H1_sit_simple.h"

#include <string>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"


namespace mjpc {
    std::string H1_sit_simple::XmlPath() const {
        return GetModelPath("Humanoid_Bench_H1/basic_locomotion/sit_simple/task.xml");
    }

    std::string H1_sit_simple::Name() const { return "H1 Sit Simple"; }

// ----------------- Residuals for Humanoid_Bench_H1 Sit Simple task ----------------

// -------------------------------------------------------------
    void H1_sit_simple::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double reward = 1.0; //TODO implement reward function
        residual[0] = 1 - reward;
    }
}  // namespace mjpc
