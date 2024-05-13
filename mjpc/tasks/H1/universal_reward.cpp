//
// Created by Moritz Meser on 11.05.24.
//

#include "universal_reward.h"

namespace mjpc {
    double calculateReward(const mjModel *model,
                           const mjData *data,
                           double vel_margin,
                           double vel_bound,
                           double hand_vel_margin,
                           double hand_vel_bound) {
        // this reward is by myself and not part of the original task
        // the main idea is to punish unrealistic movements of the robot

        double reward = 1.0;

        // ----- hand height ----- //
        // idea: limit the height of the hands, as it is not that natural to have them above the head
        double right_hand_height = SensorByName(model, data, "right_hand_position")[2];
        double left_hand_height = SensorByName(model, data, "left_hand_position")[2];
        double right_hand_reward = tolerance(right_hand_height, {0.0, 1.8}, 0.1);  // not much above the head height
        double left_hand_reward = tolerance(left_hand_height, {0.0, 1.8}, 0.1);  // not much above the head height
        double hand_reward = (right_hand_reward + left_hand_reward) / 2;
        reward *= hand_reward;


        // ----- hand velocity ----- //
        // idea: limit the velocity of the hands in cartesian space, as they tend to move around very fast
        double *right_hand_velocity = SensorByName(model, data, "right_hand_velocity");
        double *left_hand_velocity = SensorByName(model, data, "left_hand_velocity");
        double right_hand_speed = std::sqrt(right_hand_velocity[0] * right_hand_velocity[0] +
                                            right_hand_velocity[1] * right_hand_velocity[1] +
                                            right_hand_velocity[2] * right_hand_velocity[2]);
        double left_hand_speed = std::sqrt(left_hand_velocity[0] * left_hand_velocity[0] +
                                           left_hand_velocity[1] * left_hand_velocity[1] +
                                           left_hand_velocity[2] * left_hand_velocity[2]);
        reward *= tolerance(right_hand_speed, {0.0, hand_vel_bound}, hand_vel_margin, "linear", 0.0);
        reward *= tolerance(left_hand_speed, {0.0, hand_vel_bound}, hand_vel_margin, "linear", 0.0);


        // ----- actuator velocity ----- //
        // idea: limit the velocity of the actuators in joint space
        double vel_reward = 1.0;
        for (int i = 0; i < model->nu; i++) {
            vel_reward *= tolerance(data->actuator_velocity[i], {-vel_bound, +vel_bound}, vel_margin, "quadratic", 0.0);
        }
        reward *= vel_reward;

        //----- foot height ----- //
        // idea: force the robot to keep the feet near the ground
        double right_foot_height = SensorByName(model, data, "right_foot_height")[2];
        double left_foot_height = SensorByName(model, data, "left_foot_height")[2];
        double right_foot_reward = tolerance(right_foot_height, {0.0, 0.1}, 0.1);
        double left_foot_reward = tolerance(left_foot_height, {0.0, 0.1}, 0.1);
        double foot_reward = (right_foot_reward + left_foot_reward) / 2;

        reward *= foot_reward;

////        // ----- Center of mass acceleration ----- //
//// TODO this make it worse
//        // idea: limit the acceleration of the center of mass
//        double *com_acceleration = SensorByName(model, data, "com_acc");
//        double acc_reward = 1.0;
//        for (int i = 0; i < 3; i++) {
//            acc_reward *= tolerance(com_acceleration[i], {-10.0, 10.0}, 0.1, "quadratic",
//                                    0.0); //TODO values are arbitrary
//        }
//
//        reward *= acc_reward;

        return reward;
    }
}