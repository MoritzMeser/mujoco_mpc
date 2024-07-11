#ifndef MUJOCO_MPC_PUSH_H
#define MUJOCO_MPC_PUSH_H

#include <string>
#include <random>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class push : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const push *task) : mjpc::BaseResidualFn(task),
                                                    task_(const_cast<push *>(task)) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            push *task_;

        };

        push() : residual_(this) {
          target_position_ = {0.85, 0.0, 1.0};
//            std::random_device rd;
//            std::mt19937 gen(rd());
//            std::uniform_real_distribution<> dis_x(0.7, 1.0);
//            std::uniform_real_distribution<> dis_y(-0.5, 0.5);
//            target_position_ = {dis_x(gen), dis_y(gen), 1.0};
////            printf("Initial target position: %f, %f\n", target_position_[0], target_position_[1]);
        }


        void TransitionLocked(mjModel *model, mjData *data) override;

        void ResetLocked(const mjModel *model) override;

    protected:

        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const

        override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual()

        override {
            return &
                    residual_;
        }

    private:
        ResidualFn residual_;
        std::array<double, 3> target_position_;
    };

    class Push_H1 : public push {
    public:
        std::string Name() const override {
            return "Push H1";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/push/Push_H1.xml");
        }
    };

    class G1_push : public push {
    public:
        std::string Name() const override {
            return "Push G1";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/push/Push_G1.xml");
        }

    };
}  // namespace mjpc

#endif //MUJOCO_MPC_PUSH_H