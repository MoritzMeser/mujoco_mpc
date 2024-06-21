#ifndef MJPC_TASKS_H1_BALANCE_SIMPLE_H_
#define MJPC_TASKS_H1_BALANCE_SIMPLE_H_

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class Balance_Simple : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const Balance_Simple *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        Balance_Simple() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class Balance_Simple_H1 : public Balance_Simple {
    public:
        std::string Name() const override {
            return "Balance H1";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/balance/Balance_H1.xml");
        }
    };
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_BALANCE_SIMPLE_H_