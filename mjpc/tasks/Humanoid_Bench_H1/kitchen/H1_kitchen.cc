//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_kitchen.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    std::string H1_kitchen::XmlPath() const {
        return GetModelPath("Humanoid_Bench_H1/kitchen/task.xml");
    }

    std::string H1_kitchen::Name() const { return "H1 Kitchen"; }

// ----------------- Residuals for Humanoid_Bench_H1 kitchen task ----------------

// -------------------------------------------------------------
    void H1_kitchen::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                          double *residual) const {
        // the reward of this task is a count of how many objects are at the target location

        double const BONUS_THRESH = 0.3;
        // ----- initialize reward ----- //
        int reward = 0;

        // initialize target positions
        std::map<std::string, std::pair<double, double>> target_positions = {
                {"bottom burner", {-0.88, -0.01}},
                {"top burner",    {-0.92, -0.01}},
                {"light switch",  {-0.69, -0.05}},
                {"slide cabinet", {0.37,  0.0}},
                {"hinge cabinet", {0.0,   1.45}},
                {"microwave",     {-0.75, 0.0}},
                {"kettle",        {-0.23, 0.75}}
        };

        bool all_completed_so_far = true;
        // ----- loop over all tasks ----- //
        for (const std::string &task: task_->tasks_to_complete_) {
            // ----- get the position of the object ----- //
            double *object_pos = SensorByName(model, data, task);
            // ----- get the position of the target ----- //
            std::pair target_location = target_positions[task];
            // ----- calculate the distance between the object and the target ----- //
            double distance = std::sqrt(std::pow(object_pos[0] - target_location.first, 2) +
                                        std::pow(object_pos[1] - target_location.second, 2));
            // ----- check if the object is at the target location ----- //
            bool completed = distance < BONUS_THRESH;
            if (completed && (all_completed_so_far || !task_->ENFORCE_TASK_ORDER)) {
                reward++;
                all_completed_so_far = all_completed_so_far && completed;
            }
        }

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for Humanoid_Bench_H1 kitchen task -------- //
// ------------------------------------------------------------ //
    void H1_kitchen::TransitionLocked(mjModel *model, mjData *data) {
        if (!REMOVE_TASKS_WHEN_COMPLETE) {
            return;
        }
        double const BONUS_THRESH = 0.3;

        // initialize target positions
        std::map<std::string, std::pair<double, double>> target_positions = {
                {"bottom burner", {-0.88, -0.01}},
                {"top burner",    {-0.92, -0.01}},
                {"light switch",  {-0.69, -0.05}},
                {"slide cabinet", {0.37,  0.0}},
                {"hinge cabinet", {0.0,   1.45}},
                {"microwave",     {-0.75, 0.0}},
                {"kettle",        {-0.23, 0.75}}
        };

        bool all_completed_so_far = true;
        std::vector<std::string> completed_tasks;
        // ----- loop over all tasks ----- //
        for (const std::string &task: tasks_to_complete_) {
            // ----- get the position of the object ----- //
            double *object_pos = SensorByName(model, data, task);
            // ----- get the position of the target ----- //
            std::pair target_location = target_positions[task];
            // ----- calculate the distance between the object and the target ----- //
            double distance = std::sqrt(std::pow(object_pos[0] - target_location.first, 2) +
                                        std::pow(object_pos[1] - target_location.second, 2));
            // ----- check if the object is at the target location ----- //
            bool completed = distance < BONUS_THRESH;
            if (completed && (all_completed_so_far || !ENFORCE_TASK_ORDER)) {
                completed_tasks.push_back(task);
                all_completed_so_far = all_completed_so_far && completed;
            }
            // remove all completed tasks
            for (const std::string &completedTask: completed_tasks) {
                tasks_to_complete_.erase(
                        std::remove(tasks_to_complete_.begin(), tasks_to_complete_.end(), completedTask),
                        tasks_to_complete_.end());
            }
        }

    }

}  // namespace mjpc
