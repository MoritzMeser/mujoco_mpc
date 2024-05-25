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

// ----------------- Residuals for Humanoid_Bench_H1 cabinet task ---------------- //
// -------------------------------------------------------------------------------- //
    void H1_cabinet::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                          double *residual) const {
        // ----- set parameters ----- //
        double standHeight = 1.65;


        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {standHeight, INFINITY}, standHeight / 4);


        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9);

        double standReward = standing * upright;


        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        double stabilizationReward = standReward * small_control;

        // ----- subtasks ----- //
        double subtaskReward = 0.0;
        bool subtaskComplete = false;

        if (task_->current_subtask_ == 1) {
            //TODO: implement subtask 1
        } else if (task_->current_subtask_ == 2) {
            //TODO: implement subtask 2
        } else if (task_->current_subtask_ == 3) {
            //TODO: implement subtask 3
        } else if (task_->current_subtask_ == 4) {
            //TODO: implement subtask 4
        } else { // all subtasks are complete
            subtaskReward = 1000.0;
            subtaskComplete = false;
        }

        // ----- reward ----- //
        double reward;
        if (task_->current_subtask_ < 5) {
            reward = 0.2 * stabilizationReward + 0.8 * subtaskReward;
        } else {
            reward = subtaskReward;
        }

        if (subtaskComplete) {
            reward += 100.0 * task_->current_subtask_;
        }

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

    // -------- Transition for Humanoid_Bench_H1 cabinet task -------- //
    // ------------------------------------------------------------ //
    void H1_cabinet::TransitionLocked(mjModel *model, mjData *data) {
        //
        //TODO: implement transition
    }

}  // namespace mjpc