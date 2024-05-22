//
// Created by Moritz Meser on 16.05.24.
//

#include "H1_hurdle.h"

#include <string>
#include <limits>
#include <cmath>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/utility/dm_control_utils_rewards.h"
#include "mjpc/tasks/Humanoid_Bench_H1/basic_locomotion/walk_reward.h"


namespace mjpc {
    std::string H1_hurdle::XmlPath() const {
        return GetModelPath("Humanoid_Bench_H1/basic_locomotion/hurdle/task.xml");
    }

    std::string H1_hurdle::Name() const { return "H1 Hurdle"; }

// ----------------- Residuals for Humanoid_Bench_H1 walk task ----------------

// -------------------------------------------------------------
    void H1_hurdle::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double const move_speed = 5.0;
        double const stand_height = 1.65;


        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {stand_height, INFINITY}, stand_height / 4);


        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.8, INFINITY}, 1.9);

        double stand_reward = standing * upright;


        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;


        // ----- move speed ----- //
        double com_velocity = SensorByName(model, data, "center_of_mass_velocity")[0];
        double move = tolerance(com_velocity, {move_speed, INFINITY}, move_speed, "linear", 0.0);
        move = (5 * move + 1) / 6;

        // ---- wall collision discount ---- //
        double wall_collision_discount = 1;

        std::vector<std::string> body_names = {"left_barrier_collision", "right_barrier_collision",
                                               "behind_barrier_collision"};

        for (const auto &body_name: body_names) {
            int body_id = mj_name2id(model, mjOBJ_GEOM, body_name.c_str());

            if (CheckAnyCollision(model, data, body_id)) {
                wall_collision_discount = 0.1;
                break;
            }
        }
        // ---- reward computation ---- //
        double reward = small_control * stand_reward * move * wall_collision_discount;

        residual[0] = 1 - reward;
    }

    bool H1_hurdle::ResidualFn::CheckAnyCollision(const mjModel *model, const mjData *data, int body_id) const {
        for (int i = 0; i < data->ncon; i++) {
            if (data->contact[i].geom1 == body_id || data->contact[i].geom2 == body_id) {
                return true;
            }
        }
        return false;
    }


}  // namespace mjpc