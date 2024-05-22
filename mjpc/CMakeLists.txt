# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(threadpool STATIC)
target_sources(
        threadpool
        PUBLIC threadpool.h
        PRIVATE threadpool.cc
)
target_link_libraries(
        threadpool
        absl::base
)
target_include_directories(threadpool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(
        libmjpc STATIC
        states/state.cc
        states/state.h
        agent.cc
        agent.h
        trajectory.cc
        trajectory.h
        utilities.cc
        utilities.h
        tasks/tasks.cc
        tasks/tasks.h
        tasks/acrobot/acrobot.cc
        tasks/acrobot/acrobot.h
        tasks/allegro/allegro.cc
        tasks/allegro/allegro.h
        tasks/bimanual/handover/handover.cc
        tasks/bimanual/handover/handover.h
        tasks/bimanual/reorient/reorient.cc
        tasks/bimanual/reorient/reorient.h
        tasks/cartpole/cartpole.cc
        tasks/cartpole/cartpole.h
        tasks/fingers/fingers.cc
        tasks/fingers/fingers.h
        tasks/humanoid/stand/stand.cc
        tasks/humanoid/stand/stand.h
        tasks/humanoid/tracking/tracking.cc
        tasks/humanoid/tracking/tracking.h
        tasks/humanoid/walk/walk.cc
        tasks/humanoid/walk/walk.h
        tasks/manipulation/common.cc
        tasks/manipulation/common.h
        tasks/manipulation/manipulation.cc
        tasks/manipulation/manipulation.h
        tasks/op3/stand.cc
        tasks/op3/stand.h
        tasks/panda/panda.cc
        tasks/panda/panda.h
        tasks/particle/particle.cc
        tasks/particle/particle.h
        tasks/rubik/solve.cc
        tasks/rubik/solve.h
        tasks/shadow_reorient/hand.cc
        tasks/shadow_reorient/hand.h
        tasks/quadrotor/quadrotor.cc
        tasks/quadrotor/quadrotor.h
        tasks/quadruped/quadruped.cc
        tasks/quadruped/quadruped.h
        tasks/swimmer/swimmer.cc
        tasks/swimmer/swimmer.h
        tasks/walker/walker.cc
        tasks/walker/walker.h
        tasks/Humanoid_Bench_H1/basic_locomotion/walk/H1_walk.cc
        tasks/Humanoid_Bench_H1/basic_locomotion/walk/H1_walk.h
        planners/planner.cc
        planners/planner.h
        planners/policy.h
        planners/include.cc
        planners/include.h
        planners/cost_derivatives.cc
        planners/cost_derivatives.h
        planners/model_derivatives.cc
        planners/model_derivatives.h
        planners/cross_entropy/planner.cc
        planners/cross_entropy/planner.h
        planners/robust/robust_planner.cc
        planners/robust/robust_planner.h
        planners/sample_gradient/planner.cc
        planners/sample_gradient/planner.h
        planners/sampling/planner.cc
        planners/sampling/planner.h
        planners/sampling/policy.cc
        planners/sampling/policy.h
        planners/gradient/gradient.cc
        planners/gradient/gradient.h
        planners/gradient/planner.cc
        planners/gradient/planner.h
        planners/gradient/policy.cc
        planners/gradient/policy.h
        planners/gradient/settings.h
        planners/gradient/spline_mapping.cc
        planners/gradient/spline_mapping.h
        planners/ilqg/backward_pass.cc
        planners/ilqg/backward_pass.h
        planners/ilqg/boxqp.h
        planners/ilqg/planner.cc
        planners/ilqg/planner.h
        planners/ilqg/policy.cc
        planners/ilqg/policy.h
        planners/ilqg/settings.h
        planners/ilqs/planner.cc
        planners/ilqs/planner.h
        estimators/batch.cc
        estimators/batch.h
        estimators/estimator.h
        estimators/include.cc
        estimators/include.h
        estimators/kalman.cc
        estimators/kalman.h
        estimators/unscented.cc
        estimators/unscented.h
        direct/direct.cc
        direct/direct.h
        direct/trajectory.h
        direct/model_parameters.cc
        direct/model_parameters.h
        spline/spline.cc
        spline/spline.h
        app.cc
        app.h
        norm.cc
        norm.h
        simulate.cc
        simulate.h
        task.cc
        task.h
        $<TARGET_OBJECTS:mujoco::platform_ui_adapter>
        tasks/Humanoid_Bench_H1/basic_locomotion/walk/H1_walk.h
        utility/dm_control_utils_rewards.cpp
        tasks/Humanoid_Bench_H1/balance_simple/balance_simple.h
        tasks/Humanoid_Bench_H1/balance_simple/balance_simple.cc
        tasks/Humanoid_Bench_H1/package/package.h
        tasks/Humanoid_Bench_H1/package/package.cc
        tasks/Humanoid_Bench_H1/reach/H1_reach.cc
        tasks/Humanoid_Bench_H1/basic_locomotion/walk_reward.cc
        tasks/Humanoid_Bench_H1/basic_locomotion/walk_reward.h
        tasks/Humanoid_Bench_H1/basic_locomotion/stand/H1_stand.cc
        tasks/Humanoid_Bench_H1/basic_locomotion/stand/H1_stand.h
        tasks/Humanoid_Bench_H1/basic_locomotion/run/H1_run.cc
        tasks/Humanoid_Bench_H1/basic_locomotion/run/H1_run.h
        tasks/Humanoid_Bench_H1/basic_locomotion/stairs/H1_stairs.cc
        tasks/Humanoid_Bench_H1/basic_locomotion/stairs/H1_stairs.h
        tasks/Humanoid_Bench_H1/basic_locomotion/climbing_upwards_reward.cc
        tasks/Humanoid_Bench_H1/basic_locomotion/climbing_upwards_reward.h
        tasks/Humanoid_Bench_H1/basic_locomotion/slide/H1_slide.cc
        tasks/Humanoid_Bench_H1/basic_locomotion/slide/H1_slide.h
        tasks/Humanoid_Bench_H1/basic_locomotion/crawl/H1_crawl.h
        tasks/Humanoid_Bench_H1/basic_locomotion/crawl/H1_crawl.cc
        tasks/Humanoid_Bench_H1/basic_locomotion/crawl/H1_crawl.h
        tasks/Humanoid_Bench_H1/basic_locomotion/hurdle/H1_hurdle.cc
        tasks/Humanoid_Bench_H1/basic_locomotion/hurdle/H1_hurdle.h
        tasks/Humanoid_Bench_H1/maze/H1_maze.cc
        tasks/Humanoid_Bench_H1/maze/H1_maze.h
        tasks/Humanoid_Bench_H1/poles/H1_poles.cc
        tasks/Humanoid_Bench_H1/poles/H1_poles.h
        tasks/Humanoid_Bench_H1/basic_locomotion/sit_simple/H1_sit_simple.cc
        tasks/Humanoid_Bench_H1/basic_locomotion/sit_simple/H1_sit_simple.h
        tasks/Humanoid_Bench_H1/push/H1_push.cc
        tasks/Humanoid_Bench_H1/push/H1_push.h
        tasks/Humanoid_Bench_H1/cabinet/H1_cabinet.cc
        tasks/Humanoid_Bench_H1/cabinet/H1_cabinet.h
        tasks/Humanoid_Bench_H1/high_bar/H1_highbar_simple.cc
        tasks/Humanoid_Bench_H1/high_bar/H1_highbar_simple.h
        tasks/Humanoid_Bench_H1/door/H1_door.cc
        tasks/Humanoid_Bench_H1/door/H1_door.h
        tasks/Humanoid_Bench_H1/truck/H1_truck.cc
        tasks/Humanoid_Bench_H1/truck/H1_truck.h
        tasks/Humanoid_Bench_H1/cube/H1_cube.cc
        tasks/Humanoid_Bench_H1/cube/H1_cube.h
        tasks/Humanoid_Bench_H1/bookshelf/H1_bookshelf.cc
        tasks/Humanoid_Bench_H1/bookshelf/H1_bookshelf.h
        tasks/Humanoid_Bench_H1/basketball/H1_basketball.cc
        tasks/Humanoid_Bench_H1/basketball/H1_basketball.h
        tasks/Humanoid_Bench_H1/window/H1_window.cc
        tasks/Humanoid_Bench_H1/window/H1_window.h
        tasks/Humanoid_Bench_H1/spoon/H1_spoon.cc
        tasks/Humanoid_Bench_H1/spoon/H1_spoon.h
        tasks/Humanoid_Bench_H1/kitchen/H1_kitchen.cc
        tasks/Humanoid_Bench_H1/kitchen/H1_kitchen.h
        tasks/Humanoid_Bench_H1/powerlift/H1_powerlift.cc
        tasks/Humanoid_Bench_H1/powerlift/H1_powerlift.cc
        tasks/Humanoid_Bench_H1/powerlift/H1_powerlift.h
        tasks/Humanoid_Bench_H1/room/H1_room.cc
        tasks/Humanoid_Bench_H1/room/H1_room.h
        tasks/Humanoid_Bench_H1/insert/H1_insert.cc
        tasks/Humanoid_Bench_H1/insert/H1_insert.h
        utility/utility_functions.cpp
        utility/utility_functions.h
)
set_target_properties(libmjpc PROPERTIES OUTPUT_NAME mjpc)
target_compile_options(libmjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_compile_definitions(libmjpc PRIVATE MJSIMULATE_STATIC)
target_link_libraries(
        libmjpc
        absl::any_invocable
        absl::check
        absl::flat_hash_map
        absl::log
        absl::random_random
        absl::span
        glfw
        lodepng
        mujoco::mujoco
        mujoco::platform_ui_adapter
        threadpool
        Threads::Threads
)
target_include_directories(libmjpc
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_BINARY_DIR}/..
)

add_executable(
        mjpc
        main.cc
)
target_link_libraries(
        mjpc
        absl::flags
        absl::flags_parse
        absl::random_random
        absl::strings
        libmjpc
        mujoco::mujoco
        threadpool
        Threads::Threads
)
target_include_directories(mjpc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(mjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_options(mjpc PRIVATE ${MJPC_LINK_OPTIONS})
target_compile_definitions(mjpc PRIVATE MJSIMULATE_STATIC)
if (APPLE)
    target_sources(mjpc PRIVATE macos_gui.mm)
    target_link_libraries(mjpc "-framework Cocoa")
endif ()

add_library(
        libtestspeed STATIC
        testspeed_app.cc
        testspeed.h
        testspeed.cc
)
target_link_libraries(
        libtestspeed
        absl::random_random
        absl::strings
        libmjpc
        mujoco::mujoco
        threadpool
        Threads::Threads
)
target_include_directories(libtestspeed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(
        testspeed
        testspeed_app.cc
)
target_link_libraries(
        testspeed
        libtestspeed
        absl::flags
        absl::flags_parse
)
target_include_directories(testspeed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(testspeed PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_options(testspeed PRIVATE ${MJPC_LINK_OPTIONS})
target_compile_definitions(testspeed PRIVATE MJSIMULATE_STATIC)

add_subdirectory(tasks)

if (BUILD_TESTING AND MJPC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()

if (MJPC_BUILD_GRPC_SERVICE)
    add_subdirectory(grpc)
endif ()
