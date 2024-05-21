//
// Created by Moritz Meser on 21.05.24.
//

#ifndef MUJOCO_MPC_H1_SPOON_H
#define MUJOCO_MPC_H1_SPOON_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    class H1_spoon : public Task {
    public:
        std::string Name() const override;

        std::string XmlPath() const override;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_spoon *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_spoon() : residual_(this) {}

// -------- Transition for Humanoid_Bench_H1 spoon task -------- //
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


#endif //MUJOCO_MPC_H1_SPOON_H
