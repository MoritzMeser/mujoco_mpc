#ifndef MJPC_TASKS_H1_REACH_H_
#define MJPC_TASKS_H1_REACH_H_

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"
#include <random>
#include <string>

namespace mjpc {
class Reach : public Task {
public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
  public:
    explicit ResidualFn(const Reach *task)
        : mjpc::BaseResidualFn(task), task_(const_cast<Reach *>(task)) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

  private:
    Reach *task_;
  };

  Reach() : residual_(this), reward_state_("walk") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-2.0, 2.0);
    std::uniform_real_distribution<> dis_last(0.2, 2.0);
    for (size_t i = 0; i < target_position_.size(); ++i) {
      if (i == target_position_.size() - 1) {
        target_position_[i] = dis_last(gen);
      } else {
        target_position_[i] = dis(gen);
      }
    }
    printf("Target position Reach: %f %f %f\n", target_position_[0],
           target_position_[1], target_position_[2]);
  }
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
  std::string Name() const override { return "Reach H1"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/reach/Reach_H1.xml");
  }
};
class Reach_G1 : public Reach {
public:
  std::string Name() const override { return "Reach G1"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/reach/Reach_G1.xml");
  }
};
} // namespace mjpc

#endif // MJPC_TASKS_H1_REACH_H_