//
// Created by Moritz Meser on 15.05.24.
//

#ifndef MJPC_TASKS_H1_RUN_H_
#define MJPC_TASKS_H1_RUN_H_

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_run : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_run *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_run() : residual_(this) {}


    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class Run_H1 : public H1_run {
    public:
        std::string Name() const override {
            return "Run H1";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/run/Run_H1.xml");
        }
    };

    class Run_G1 : public H1_run {
    public:
        std::string Name() const override {
            return "Run G1";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/run/Run_G1.xml");
        }
    };
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_RUN_H_