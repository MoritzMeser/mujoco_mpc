//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_cabinet.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    std::string H1_cabinet::XmlPath() const {
        return GetModelPath("Humanoid_Bench_H1/cabinet/task.xml");
    }

    std::string H1_cabinet::Name() const { return "H1 Cabinet"; }

// ----------------- Residuals for Humanoid_Bench_H1 cabinet task ----------------

// -------------------------------------------------------------
    void H1_cabinet::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                        double *residual) const {
        //TODO: implement this
        double reward = 1.0;

        printf("Residuals, current_subtask = %d", task_->current_subtask_);
        // ----- residuals ----- //

        residual[0] = std::exp(-reward);
    }

// -------- Transition for Humanoid_Bench_H1 cabinet task -------- //
// ------------------------------------------------------------ //
    void H1_cabinet::TransitionLocked(mjModel *model, mjData *data) {
        //
        printf("TransitionLocked, current_subtask = %d", current_subtask_);
    }

}  // namespace mjpc