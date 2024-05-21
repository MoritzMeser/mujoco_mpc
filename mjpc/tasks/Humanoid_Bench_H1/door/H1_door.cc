//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_door.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    std::string H1_door::XmlPath() const {
        return GetModelPath("Humanoid_Bench_H1/door/task.xml");
    }

    std::string H1_door::Name() const { return "H1 Door"; }

// ----------------- Residuals for Humanoid_Bench_H1 door task ----------------

// -------------------------------------------------------------
    void H1_door::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                       double *residual) const {
        //TODO: implement this
        double reward = 1.0;

        // ----- residuals ----- //

        residual[0] = std::exp(-reward);
    }

// -------- Transition for Humanoid_Bench_H1 door task -------- //
// ------------------------------------------------------------ //
    void H1_door::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
