//
// Created by Moritz Meser on 15.05.24.
//

#ifndef MJPC_TASKS_H1_WALK_H_
#define MJPC_TASKS_H1_WALK_H_

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"


namespace mjpc {
    class Walk : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const Walk *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        Walk() : residual_(this) {}


    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class Walk_H1 : public Walk {
    public:
        std::string Name() const override {
            return "Walk H1";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/Walk/Walk_H1.xml");
        }
    };

    class Walk_G1 : public Walk {
    public:
        std::string Name() const override {
            return "Walk G1";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/Walk/Walk_G1.xml");
        }
    };

    class Walk_H1_Torque : public Walk {
    public:
        std::string Name() const override {
            return "Walk H1 Torque Control";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/Walk/Walk_H1_Torque.xml");
        }
    };
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_WALK_H_
