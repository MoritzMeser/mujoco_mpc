//
// Created by Moritz Meser on 11.05.24.
//

#ifndef MUJOCO_MPC_UNIVERSAL_REWARD_H
#define MUJOCO_MPC_UNIVERSAL_REWARD_H

#include <string>
#include <limits>
#include <cmath>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    double calculateReward(const mjModel *model, const mjData *data, double vel_margin, double vel_bound);
}

#endif //MUJOCO_MPC_UNIVERSAL_REWARD_H
