//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_push.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    std::string H1_push::XmlPath() const {
        return GetModelPath("Humanoid_Bench_H1/push/task.xml");
    }

    std::string H1_push::Name() const { return "H1 Push"; }

// ----------------- Residuals for Humanoid_Bench_H1 push task ----------------

// -------------------------------------------------------------
    void H1_push::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                        double *residual) const {
        //TODO: implement this
        double reward = 1.0;

        // ----- residuals ----- //

        residual[0] = std::exp(-reward);
    }

// -------- Transition for Humanoid_Bench_H1 push task -------- //
// ------------------------------------------------------------ //
    void H1_push::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc