#ifndef MJPC_TASKS_H1_REACH_H_
#define MJPC_TASKS_H1_REACH_H_

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class Reach : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const Reach *task) : mjpc::BaseResidualFn(task),
                                                     task_(const_cast<Reach *>(task)) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            Reach *task_;
        };

        Reach() : residual_(this), target_position_({1.0, 1.0, 1.0}), reward_state_("walk") {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
        std::array<double, 3> target_position_;
        std::string reward_state_;

    };

    class Reach_H1 : public Reach {
    public:
        std::string Name() const override {
            return "Reach H1";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/reach/Reach_H1.xml");
        }
    };
    class Reach_G1 : public Reach {
    public:
        std::string Name() const override {
            return "Reach G1";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/reach/Reach_G1.xml");
        }
    };
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_REACH_H_