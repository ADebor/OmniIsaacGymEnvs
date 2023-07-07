# Tasks

This folder contains the definition of several robotic tasks. In particular, `local_bm.py` implements the "one-finger", local-control task, *aka* the "local benchmark", and `global_bm.py` implements the "full-hand", global-control task, *aka* the "global benchmark".

## Local benchmark

The local benchmark consists of a simple task involving only one robotic finger. The latter is effort controlled, and interacts with a so-called "sliding button". The artificial agent controlling the robotic finger receives pressure data from sensors located on the fingertip. The goal for the agent is to lift the button by pressing with the finger so as to place it in the high-reward zone. 

We consider two ways of making the task a benchmark for assessing the adaptation capabilities of the artificial agent controlling the finger:
- The position and length of the high-reward zone can change from one episode to another.
- The physical properties of the button (stiffness,  damping, ...) can change from one episode to another.

<p align="center">
  <img src="images/local_bm_img.jpg" />
</p>


## Global benchmark

The global benchmark consists of an adaptation of the [dexterous manipulation](https://arxiv.org/abs/1808.00177) benchmark developed by OpenAI, in which the robotic hand has to orient a cubic block in the same way as a goal cube. In our version, the artificial agent controls the hand using effort (force) control instead of position control.

The forked repository proposes 4 different types of observations (for the policy network), all of them considering "extrinsic"[^1] sensing:

- "`openai`": fingertip positions (15), object position (3), relative target orientation (4) (and actions taken (20))
- "`full_no_vel`": fingertip positions (15), dof positions (hand joints angles) (24), object position (3), object orientation (4), target position (3), target orientation (4), relative target orientation (4) (and actions taken (20))
- "`full`": same as "`full_no_vel`" + hand joints velocities (24), object velocity (3), object angular velocity (3), fingertip orientations (20), fingertip velocities (5 * ( 3 linear + 3 angular ) = 30) 
- "`full_state`": same as "`full`" + torque sensing data (30) (joint torque sensors not working in Isaac sim, still)
  
We add 11 more types that use "intrinsic"[^1] sensing:

- "`intrinsic_openai`": same as "`openai`" + pressure/tactile sensing data added:
    - one resulting force scalar per fingertip (5), or
    - one force, position, and orientation per contact point (varying size)  (to be implemented)
- "`intrinsic_openai_strict`": same as "`intrinsic_openai`" - fingertip positions (15)
- "`intrinsic_full_no_vel`": same as "`full_no_vel`" + pressure/tactile sensing data added
- "`intrinsic_full_no_vel_strict`": same as "`intrinsic_full_no_vel`" - fingertip positions (15)
- "`intrinsic_full_no_vel_strict_no_proprio`": same as "`intrinsic_full_no_vel_strict`" - dof positions (24) 
- "`intrinsic_full`": same as "`full`" + pressure/tactile sensing data added
- "`intrinsic_full_strict`": same as "`intrinsic_full`" - fingertip positions (15) - fingertip orientations (20) - fingertip velocities (30)
- "`intrinsic_full_strict_no_proprio`": same as "`intrinsic_full_strict`" - dof positions (24) - dof velocities (24)
- "`intrinsic_full_state`": same as "`full_state`" + pressure/tactile sensing data added
- "`intrinsic_full_state_strict`": same as "`intrinsic_full_state`" - fingertip positions (15) - fingertip orientations (20) - fingertip velocities (30)
- "`intrinsic_full_state_strict_no_proprio`": same as "`intrinsic_full_state_strict`" - dof positions (24) - dof velocities (24) - torque sensing data (30)

We further add a "handicap" feature that mimics arthrosis by (to be defined).

<p align="center">
  <img src="https://user-images.githubusercontent.com/34286328/171454160-8cb6739d-162a-4c84-922d-cda04382633f.gif" width="300" height="150"/>
</p>

[^1]: We refer to [this](https://sbrl.cs.columbia.edu/) for the notions of "intrinsic" and "extrinsic" sensing, which do not really make sense in the context of simulation. As we add tactile data though, we use these terms to discriminate between the different observation types. Note: in [this](https://sbrl.cs.columbia.edu/), the observations are [dof positions, dof setpoints, binary contacts] for the actor, and the same as for the actor + [object position, object velocity, net contact force for each finger] for the critic.
