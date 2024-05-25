//
// Created by Moritz Meser on 21.05.24.
//

#ifndef MUJOCO_MPC_H1_TRUCK_H
#define MUJOCO_MPC_H1_TRUCK_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utility/dm_control_utils_rewards.h"

namespace mjpc {
    class H1_truck : public Task {
    public:
        bool IsPackageUponTable(const mjModel *model, const mjData *data, const std::string &package) const;

        std::string Name() const override;

        std::string XmlPath() const override;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_truck *task) : mjpc::BaseResidualFn(task),
                                                        task_(const_cast<H1_truck *>(task)) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            H1_truck *task_;
        };

        H1_truck() : residual_(this),
                     initial_zs_({}),
                     packages_on_truck_({"package_a", "package_b", "package_c", "package_d", "package_e"}),
                     packages_picked_up_({}),
                     packages_upon_table_({}) {}

// -------- Transition for Humanoid_Bench_H1 truck task -------- //
// ------------------------------------------------------------ //
        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
        std::vector<double> initial_zs_;
        std::vector<std::string> packages_on_truck_;
        std::vector<std::string> packages_picked_up_;
        std::vector<std::string> packages_upon_table_;
    };
}  // namespace mjpc


#endif //MUJOCO_MPC_H1_TRUCK_H
