//
// Created by Moritz Meser on 15.05.24.
//

#ifndef MJPC_TASKS_H1_STAND_H_
#define MJPC_TASKS_H1_STAND_H_

#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
class Stand : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Stand *task) : mjpc::BaseResidualFn(task) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;
  };

  Stand() : residual_(this) {}

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn *InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

class Stand_H1 : public Stand {
 public:
  std::string Name() const override { return "Stand H1"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/stand/Stand_H1.xml");
  }
};

class Stand_G1 : public Stand {
 public:
  std::string Name() const override { return "Stand G1"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/stand/Stand_G1.xml");
  }
};
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_STAND_H_