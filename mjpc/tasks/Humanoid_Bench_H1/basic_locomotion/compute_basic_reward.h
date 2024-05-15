//
// Created by Moritz Meser on 15.05.24.
//

#ifndef MUJOCO_MPC_COMPUTE_BASIC_REWARD_H
#define MUJOCO_MPC_COMPUTE_BASIC_REWARD_H

#include "mujoco/mujoco.h"

namespace mjpc {
    double compute_basic_reward(const mjModel *model, const mjData *data, double walk_speed, double stand_height);
} // namespace mjpc
#endif //MUJOCO_MPC_COMPUTE_BASIC_REWARD_H
