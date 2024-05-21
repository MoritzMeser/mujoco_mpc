// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/tasks.h"

#include <memory>
#include <vector>

#include "mjpc/task.h"
#include "mjpc/tasks/acrobot/acrobot.h"
#include "mjpc/tasks/allegro/allegro.h"
#include "mjpc/tasks/bimanual/handover/handover.h"
#include "mjpc/tasks/bimanual/reorient/reorient.h"
#include "mjpc/tasks/cartpole/cartpole.h"
#include "mjpc/tasks/fingers/fingers.h"
#include "mjpc/tasks/humanoid/stand/stand.h"
#include "mjpc/tasks/humanoid/tracking/tracking.h"
#include "mjpc/tasks/humanoid/walk/walk.h"
#include "mjpc/tasks/manipulation/manipulation.h"
// DEEPMIND INTERNAL IMPORT
#include "mjpc/tasks/op3/stand.h"
#include "mjpc/tasks/panda/panda.h"
#include "mjpc/tasks/particle/particle.h"
#include "mjpc/tasks/quadrotor/quadrotor.h"
#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/tasks/rubik/solve.h"
#include "mjpc/tasks/shadow_reorient/hand.h"
#include "mjpc/tasks/swimmer/swimmer.h"
#include "mjpc/tasks/walker/walker.h"
// Humanoid Bench Tasks
#include "mjpc/tasks/Humanoid_Bench_H1/basic_locomotion/walk/H1_walk.h"
#include "mjpc/tasks/Humanoid_Bench_H1/basic_locomotion/stand/H1_stand.h"
#include "mjpc/tasks/Humanoid_Bench_H1/basic_locomotion/run/H1_run.h"
#include "mjpc/tasks/Humanoid_Bench_H1/basic_locomotion/stairs/H1_stairs.h"
#include "mjpc/tasks/Humanoid_Bench_H1/basic_locomotion/slide/H1_slide.h"
#include "mjpc/tasks/Humanoid_Bench_H1/basic_locomotion/crawl/H1_crawl.h"
#include "mjpc/tasks/Humanoid_Bench_H1/basic_locomotion/hurdle/H1_hurdle.h"
#include "mjpc/tasks/Humanoid_Bench_H1/basic_locomotion/sit_simple/H1_sit_simple.h"
#include "mjpc/tasks/Humanoid_Bench_H1/maze/H1_maze.h"
#include "mjpc/tasks/Humanoid_Bench_H1/poles/H1_poles.h"
#include "mjpc/tasks/Humanoid_Bench_H1/push/H1_push.h"
#include "mjpc/tasks/Humanoid_Bench_H1/cabinet/H1_cabinet.h"
#include "mjpc/tasks/Humanoid_Bench_H1/high_bar/H1_highbar_simple.h"
#include "mjpc/tasks/Humanoid_Bench_H1/door/H1_door.h"
#include "mjpc/tasks/Humanoid_Bench_H1/truck/H1_truck.h"
#include "mjpc/tasks/Humanoid_Bench_H1/cube/H1_cube.h"
#include "mjpc/tasks/Humanoid_Bench_H1/bookshelf/H1_bookshelf.h"
#include "mjpc/tasks/Humanoid_Bench_H1/basketball/H1_basketball.h"
#include "mjpc/tasks/Humanoid_Bench_H1/window/H1_window.h"
#include "mjpc/tasks/Humanoid_Bench_H1/spoon/H1_spoon.h"
#include "mjpc/tasks/Humanoid_Bench_H1/kitchen/H1_kitchen.h"
#include "mjpc/tasks/Humanoid_Bench_H1/package/package.h"
#include "mjpc/tasks/Humanoid_Bench_H1/powerlift/H1_powerlift.h"
#include "mjpc/tasks/Humanoid_Bench_H1/room/H1_room.h"
#include "mjpc/tasks/Humanoid_Bench_H1/insert/H1_insert.h"


#include "mjpc/tasks/Humanoid_Bench_H1/balance_simple/balance_simple.h"

#include "mjpc/tasks/Humanoid_Bench_H1/reach/H1_reach.h"

namespace mjpc {

    std::vector<std::shared_ptr<Task>> GetTasks() {
        return {
                //Humanoid Bench Tasks
                std::make_shared<H1_walk>(),
                std::make_shared<H1_stand>(),
                std::make_shared<H1_run>(),
                std::make_shared<H1_stairs>(),
                std::make_shared<H1_slide>(),
                std::make_shared<H1_crawl>(),
                std::make_shared<H1_hurdle>(),
                std::make_shared<H1_maze>(),
                std::make_shared<H1_poles>(),
                std::make_shared<H1_sit_simple>(),
                std::make_shared<H1_push>(),
                std::make_shared<H1_cabinet>(),
                std::make_shared<H1_highbar>(),
                std::make_shared<H1_door>(),
                std::make_shared<H1_truck>(),
                std::make_shared<H1_cube>(),
                std::make_shared<H1_bookshelf>(),
                std::make_shared<H1_basketball>(),
                std::make_shared<H1_window>(),
                std::make_shared<H1_spoon>(),
//                std::make_shared<H1_kitchen>(), //TODO: Fix H1_kitchen, currently not working
                std::make_shared<H1_package>(),
                std::make_shared<H1_powerlift>(),
                std::make_shared<H1_room>(),
                std::make_shared<H1_insert>(),


//                std::make_shared<Balance_Simple>(),

//                std::make_shared<H1_reach>(),
//                // other tasks
//                std::make_shared<Acrobot>(),
//                std::make_shared<Allegro>(),
//                std::make_shared<aloha::Handover>(),
//                std::make_shared<aloha::Reorient>(),
//                std::make_shared<Cartpole>(),
//                std::make_shared<Fingers>(),
//                std::make_shared<humanoid::Stand>(),
//                std::make_shared<humanoid::Tracking>(),
//                std::make_shared<humanoid::Walk>(),
//                std::make_shared<manipulation::Bring>(),
//                // DEEPMIND INTERNAL TASKS
//                std::make_shared<OP3>(),
//                std::make_shared<Panda>(),
//                std::make_shared<Particle>(),
//                std::make_shared<ParticleFixed>(),
//                std::make_shared<Rubik>(),
//                std::make_shared<ShadowReorient>(),
//                std::make_shared<Quadrotor>(),
//                std::make_shared<QuadrupedFlat>(),
//                std::make_shared<QuadrupedHill>(),
//                std::make_shared<Swimmer>(),
//                std::make_shared<Walker>(),
        };
    }
}  // namespace mjpc
