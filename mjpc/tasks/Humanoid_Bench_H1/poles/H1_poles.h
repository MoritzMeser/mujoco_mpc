//
// Created by Moritz Meser on 20.05.24.
//

#ifndef MUJOCO_MPC_H1_POLES_H
#define MUJOCO_MPC_H1_POLES_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    class H1_poles : public Task {
    public:
        std::string Name() const override;

        std::string XmlPath() const override;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_poles *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_poles() : residual_(this) {}

// -------- Transition for Humanoid_Bench_H1 poles task -------- //
// ------------------------------------------------------------ //
        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };
}  // namespace mjpc


#endif //MUJOCO_MPC_H1_POLES_H
