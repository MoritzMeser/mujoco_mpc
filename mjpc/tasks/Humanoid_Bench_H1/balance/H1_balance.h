#ifndef MJPC_TASKS_H1_Balance_Simple_H_
#define MJPC_TASKS_H1_Balance_Simple_H_

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"

namespace mjpc {
    class Balance_Simple : public Task {
    public:
        std::string Name() const override;

        std::string XmlPath() const override;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const Balance_Simple *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        Balance_Simple() : residual_(this) {}

// -------- Transition for Humanoid_Bench_H1 walk task --------
//  for a more complex task this might be necessary (like walking to different targets)
// ---------------------------------------------
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

#endif  // MJPC_TASKS_H1_Balance_Simple_H_
