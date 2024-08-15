//
// Created by Moritz Meser on 22.05.24.
//

#include "utility_functions.h"

#include "mujoco/mujoco.h"

bool CheckAnyCollision(const mjModel *model, const mjData *data, int body_id) {
  for (int i = 0; i < data->ncon; i++) {
    if (data->contact[i].geom1 == body_id ||
        data->contact[i].geom2 == body_id) {
      return true;
    }
  }
  return false;
}

bool CheckBodyCollision(const mjModel *model, const mjData *data,
                        const char *body_1_name, const char *body_2_name) {
  int body_1_id = mj_name2id(model, mjOBJ_GEOM, body_1_name);
  int body_2_id = mj_name2id(model, mjOBJ_GEOM, body_2_name);

  if (body_1_id < 0 || body_2_id < 0) {
    return false;
  }

  for (int i = 0; i < data->ncon; i++) {
    if (data->contact[i].geom1 == body_1_id &&
        data->contact[i].geom2 == body_2_id) {
      return true;
    }
    if (data->contact[i].geom1 == body_2_id &&
        data->contact[i].geom2 == body_1_id) {
      return true;
    }
  }

  return false;
}
