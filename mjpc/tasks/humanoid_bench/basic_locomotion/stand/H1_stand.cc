//
// Created by Moritz Meser on 15.05.24.
//

#include "H1_stand.h"

#include <string>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/basic_locomotion/walk_reward.h"
#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"


namespace mjpc {
// ----------------- Residuals for humanoid_bench stand task ---------------- //
// ----------------------------------------------------------------------------- //
    void H1_stand::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double const walk_speed = 0.0;
        double const stand_height = 1.65;

        int residual_counter = 0;
        // ----- original humanoid bench reward ----- //
        double reward = walk_reward(model, data, walk_speed, stand_height);
        residual[residual_counter++] = std::exp(-reward);

        // ----- individual reward terms ----- //

        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {stand_height, INFINITY}, stand_height / 4);

        residual[residual_counter++] = 1.0 - standing;


        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9);

        residual[residual_counter++] = 1.0 - upright;


        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls

        residual[residual_counter++] = 1.0 - small_control;


        // ----- move speed ----- //
        double move_reward = 1.0;
        if (walk_speed == 0.0) {
            double horizontal_velocity_x = SensorByName(model, data, "center_of_mass_velocity")[0];
            double horizontal_velocity_y = SensorByName(model, data, "center_of_mass_velocity")[1];
            double dont_move = (tolerance(horizontal_velocity_x, {0.0, 0.0}, 2) +
                                tolerance(horizontal_velocity_y, {0.0, 0.0}, 2)) / 2;
            move_reward = dont_move;
        } else {
            double com_velocity = SensorByName(model, data, "center_of_mass_velocity")[0];
            double move = tolerance(com_velocity, {walk_speed, INFINITY}, std::abs(walk_speed), "linear", 0.0);
            move = (5 * move + 1) / 6;
            move_reward = move;
        }

        residual[residual_counter++] = 1.0 - move_reward;
    }
}  // namespace mjpc
