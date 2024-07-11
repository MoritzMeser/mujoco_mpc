#ifndef MJPC_TASKS_H1_PUNCH_H_
#define MJPC_TASKS_H1_PUNCH_H_

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class Punch : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const Punch *task) : mjpc::BaseResidualFn(task),
                                                     task_(const_cast<Punch *>(task)) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            Punch *task_;
        };

        Punch() : residual_(this), target_position_({0.4, 0.1, 1.0}), use_left_hand_(true) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
        std::array<double, 3> target_position_;
        bool use_left_hand_;
    };

    class Punch_H1 : public Punch {
    public:
        std::string Name() const override {
            return "Punch H1";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/punch/Punch_H1.xml");
        }
    };

    class Punch_G1 : public Punch {
    public:
        std::string Name() const override {
            return "Punch G1";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/punch/Punch_G1.xml");
        }
    };

}  // namespace mjpc

#endif  // MJPC_TASKS_H1_PUNCH_H_