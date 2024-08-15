//
// Created by Moritz Meser on 22.05.24.
//

#ifndef MUJOCO_MPC_UTILITY_FUNCTIONS_H
#define MUJOCO_MPC_UTILITY_FUNCTIONS_H

#include "mujoco/mjmodel.h"
#include "mujoco/mjdata.h"
#include <string>

bool CheckAnyCollision(const mjModel *pModel, const mjData *pData, int id);

bool CheckBodyCollision(const mjModel *pModel, const mjData *pData,
                        const char* body_1_name,  const char* body_2_name);

#endif //MUJOCO_MPC_UTILITY_FUNCTIONS_H
