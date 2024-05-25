//
// Created by Moritz Meser on 21.05.24.
//

#ifndef MUJOCO_MPC_H1_CABINET_H
#define MUJOCO_MPC_H1_CABINET_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    class H1_cabinet : public Task {
    public:
        std::string Name() const override;

        std::string XmlPath() const override;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_cabinet *task) : mjpc::BaseResidualFn(task), task_(
                    const_cast<H1_cabinet *>(task)) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            H1_cabinet *task_;
        };


        H1_cabinet() : residual_(this), current_subtask_(1) {}

// -------- Transition for Humanoid_Bench_H1 cabinet task -------- //
// ------------------------------------------------------------ //
        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
        mutable int current_subtask_;
    };
}  // namespace mjpc


#endif //MUJOCO_MPC_H1_CABINET_H
