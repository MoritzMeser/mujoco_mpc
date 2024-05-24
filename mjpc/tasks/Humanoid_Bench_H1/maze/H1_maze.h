//
// Created by Moritz Meser on 20.05.24.
//

#ifndef MUJOCO_MPC_H1_MAZE_H
#define MUJOCO_MPC_H1_MAZE_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    class H1_maze : public Task {
    public:
        std::string Name() const override;

        std::string XmlPath() const override;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_maze *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_maze() : residual_(this) {}

// -------- Transition for Humanoid_Bench_H1 Maze task -------- //
// ------------------------------------------------------------ //
        void TransitionLocked(mjModel *model, mjData *data) override;

        // call base-class Reset, save task-related ids
        void ResetLocked(const mjModel* model) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
        int curr_goal_idx_ = 0;
    };
}  // namespace mjpc


#endif //MUJOCO_MPC_H1_MAZE_H
