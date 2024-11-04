#ifndef MUJOCO_MPC_H1_DOOR_H
#define MUJOCO_MPC_H1_DOOR_H

#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
class H1_door : public Task {
public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
  public:
    explicit ResidualFn(const H1_door *task)
        : mjpc::BaseResidualFn(task), task_(const_cast<H1_door *>(task)) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

  private:
    H1_door *task_;
  };

  H1_door()
      : residual_(this),
        reward_machine_state_(1) {
  }  // TODO we start with state 1, since the handle is open currently

  void TransitionLocked(mjModel *model, mjData *data) override;

protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn *InternalResidual() override { return &residual_; }

private:
  ResidualFn residual_;
  int reward_machine_state_;
};

class H1_door_position : public H1_door {
public:
  std::string Name() const override { return "Door H1"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/door/Door_H1.xml");
  }
};

class H1_door_hand : public H1_door {
public:
  std::string Name() const override { return "H1 Door Hand"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/door/H1_door_hand.xml");
  }
};

class H1_door_gripper : public H1_door {
public:
  std::string Name() const override { return "H1 Door Gripper"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/door/H1_door_gripper.xml");
  }
};

class H1_door_simple_hand : public H1_door {
public:
  std::string Name() const override { return "H1 Door Simple Hand"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/door/H1_door_simple_hand.xml");
  }
};

class H1_door_strong : public H1_door {
public:
  std::string Name() const override { return "H1 Door Strong"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/door/H1_door_strong.xml");
  }
};

class H1_door_touch : public H1_door {
public:
  std::string Name() const override { return "H1 Door Touch"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/door/H1_door_touch.xml");
  }
};
}  // namespace mjpc

#endif  // MUJOCO_MPC_H1_DOOR_H