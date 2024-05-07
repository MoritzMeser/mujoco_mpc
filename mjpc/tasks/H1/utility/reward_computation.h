//
// Created by Moritz Meser on 26.04.24.
//

#ifndef MUJOCO_MPC_REWARD_COMPUTATION_H
#define MUJOCO_MPC_REWARD_COMPUTATION_H


#include <string>
#include <utility>
#include "mujoco/mujoco.h"

double sigmoid(double x, double value_at_1, std::string sigmoid_type);

double tolerance(double x, std::pair<double, double> bounds = {0.0, 0.0}, double margin = 0.0,
                 std::string sigmoid_type = "gaussian", double value_at_margin = 0.1);

void rotateVector(mjtNum *vector, const mjtNum *rotationMatrix, const mjtNum *inputVector);

#endif //MUJOCO_MPC_REWARD_COMPUTATION_H
