# Humanoid Bench in MuJoCo-MPC

This directory contains the re-implementation of some of the humanoid benchmark in MuJoCo-MPC. The original implementation is in [this repository](https://github.com/carlosferrazza/humanoid-bench).

## Reward to Residuals
MuJoCo-MPC uses residuals with multiple dimensions instead of a single reward.
The residuals should be 'close to zero' to indicate a good performance. So in each task, the first step is to compute the reward the same way it is done in the original implementation. 
Then, the first dimension of the residual is set to x - reward, where x is the maximum reward that can be achieved in the task.

## Additional Residuals
In addition to the reward residual, we also add additional residuals. We found them to be helpful to solve the task. 
To get the 'vanilla' version of the task, you can set the additional residuals weights to zero, using the sliders in the GUI.

## Robots
In the original implementation, they use a position controlled H1 robot from unitree. In each task, there are different versions of the robot, available. 
In Addition to the 'normal' one, also one version which is stronger, and also versions with various hands attached to the robot.

We use also the position controlled H1 robot, but the version with hands is only included in tasks, were the hands are beneficial to solve the task (i.e. tasks that are not pure locomotion).

In addition, in some of the tasks we also include the G1 robot from unitree. This robot is torque controlled.

## Punch Task
This task is not part of the original benchmark. It is a task where the robot has to punch a target. The target is a sphere, which is placed at a random position in front of the robot. The robot has to punch the target with alternating hands.
