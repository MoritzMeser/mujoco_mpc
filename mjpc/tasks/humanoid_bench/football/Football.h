#ifndef MUJOCO_MPC_H1_Football_H
#define MUJOCO_MPC_H1_Football_H
#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class Football : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const Football *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        Football() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class Football_H1 : public Football {
    public:
        std::string Name() const override {
            return "Football H1";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/football/Football_H1.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_Football_H