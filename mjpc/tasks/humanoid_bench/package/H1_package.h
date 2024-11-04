#ifndef MUJOCO_MPC_H1_PACKAGE_H
#define MUJOCO_MPC_H1_PACKAGE_H

#include <random>
#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
class H1_package : public Task {
public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
  public:
    explicit ResidualFn(const H1_package *task)
        : mjpc::BaseResidualFn(task), task_(const_cast<H1_package *>(task)) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

  private:
    const H1_package *task_;
  };

  H1_package() : residual_(this), reward_machine_state_(0) {
    //            std::random_device rd;
    //            std::mt19937 gen(rd());
    //            std::uniform_real_distribution<> dis_x(-2, 2);
    //            std::uniform_real_distribution<> dis_y(-2, 2);
    //
    //
    //            target_position_ = {dis_x(gen), dis_y(gen), 0.35};
    target_position_ = {2.0, 2.0, 0.35}, squat_ = 0.0, timestamp_ = 0.0;
    printf("Initial target position: %f, %f\n", target_position_[0],
           target_position_[1]);
  }

  void TransitionLocked(mjModel *model, mjData *data) override;

  void ResetLocked(const mjModel *model) override;

protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn *InternalResidual() override { return &residual_; }

private:
  ResidualFn residual_;
  int reward_machine_state_;
  std::array<double, 3> target_position_;
  double squat_;
  double timestamp_;
};

class Package_H1 : public H1_package {
public:
  std::string Name() const override { return "Package H1"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/package/Package_H1.xml");
  }
};
}  // namespace mjpc

#endif  // MUJOCO_MPC_H1_PACKAGE_H