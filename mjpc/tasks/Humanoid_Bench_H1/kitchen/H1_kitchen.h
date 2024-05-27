//
// Created by Moritz Meser on 21.05.24.
//

#ifndef MUJOCO_MPC_H1_KITCHEN_H
#define MUJOCO_MPC_H1_KITCHEN_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    class H1_kitchen : public Task {
    public:
        std::string Name() const override;

        std::string XmlPath() const override;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_kitchen *task) : mjpc::BaseResidualFn(task),
                                                          task_(const_cast<H1_kitchen *>(task)) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            H1_kitchen *task_;
        };

        H1_kitchen() : residual_(this),
                       REMOVE_TASKS_WHEN_COMPLETE(true),
                       TERMINATE_ON_TASK_COMPLETE(true),
                       ENFORCE_TASK_ORDER(true),
                       tasks_to_complete_({"microwave", "kettle", "bottom burner", "light switch"}) {}

// -------- Transition for Humanoid_Bench_H1 kitchen task -------- //
// ------------------------------------------------------------ //
        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
        bool REMOVE_TASKS_WHEN_COMPLETE;
        bool TERMINATE_ON_TASK_COMPLETE;
        bool ENFORCE_TASK_ORDER;
        std::vector<std::string> tasks_to_complete_;
    };
}  // namespace mjpc


#endif //MUJOCO_MPC_H1_KITCHEN_H
