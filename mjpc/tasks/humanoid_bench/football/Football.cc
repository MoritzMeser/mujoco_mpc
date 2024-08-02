//
// Created by Moritz Meser on 21.05.24.
//

#include "Football.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench basketball task
// ---------------- //
// ----------------------------------------------------------------------------------
// //
void Football::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                    double *residual) const {}

// -------- Transition for humanoid_bench basketball task -------- //
// ------------------------------------------------------------ //
void Football::TransitionLocked(mjModel *model, mjData *data) {
  //
}

}  // namespace mjpc
