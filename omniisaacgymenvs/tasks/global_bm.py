"""
TO DO
Effort control OpenAI-like task
Roadmap:
1) Effort control instead of position control (same sensing data as OpenAI) DONE
2) Intrinsic sensing instead of extrinsic sensing (touch and proprioceptive only, no vision) - still effort control DONE
3) Handicap feature added - adaptive behavior learning using neuromodulation
"""


from omniisaacgymenvs.tasks.shared.in_hand_manipulation_base import (
    InHandManipulationBaseTask,
)
from omniisaacgymenvs.robots.articulations.shadow_hand import ShadowHand
from omniisaacgymenvs.robots.articulations.views.shadow_hand_view import ShadowHandView

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch import *

import torch


class GlobalBenchmarkTask(InHandManipulationBaseTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.object_type = self._task_cfg["env"]["objectType"]
        assert self.object_type in ["block"]

        self.obs_type = self._task_cfg["env"]["observationType"]
        if not (
            self.obs_type
            in [
                "openai",
                "full_no_vel",
                "full",
                "full_state",
                "intrinsic_openai",
                "intrinsic_openai_strict",
                "intrinsic_full_no_vel",
                "intrinsic_full_no_vel_strict",
                "intrinsic_full_no_vel_strict_no_proprio",
                "intrinsic_full",
                "intrinsic_full_strict",
                "intrinsic_full_strict_no_proprio",
                "intrinsic_full_state",
                "intrinsic_full_state_strict",
                "intrinsic_full_state_strict_no_proprio",
            ]
        ):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]"
            )
        print("Obs type:", self.obs_type)
        self.obs_dict = {
            "openai": [self.get_ft_pos, self.get_obj_pos, self.get_rel_rot],
            "full_no_vel": [
                self.get_ft_pos,
                self.get_dof_pos,
                self.get_obj_pos,
                self.get_obj_rot,
                self.get_goal_pos,
                self.get_goal_rot,
                self.get_rel_rot,
            ],
            "full": [
                self.get_ft_pos,
                self.get_dof_pos,
                self.get_obj_pos,
                self.get_obj_rot,
                self.get_goal_pos,
                self.get_goal_rot,
                self.get_rel_rot,
            ]
            + [
                self.get_dof_vel,
                self.get_obj_lin_vel,
                self.get_obj_ang_vel,
                self.get_ft_rot,
                self.get_ft_vel,
            ],
            "full_state": [
                self.get_ft_pos,
                self.get_dof_pos,
                self.get_obj_pos,
                self.get_obj_rot,
                self.get_goal_pos,
                self.get_goal_rot,
                self.get_rel_rot,
            ]
            + [
                self.get_dof_vel,
                self.get_obj_lin_vel,
                self.get_obj_ang_vel,
                self.get_ft_rot,
                self.get_ft_vel,
            ]
            + [self.get_torque_obs],
            "intrinsic_openai": [self.get_ft_pos, self.get_obj_pos, self.get_rel_rot]
            + [self.get_tactile_obs],
            "intrinsic_openai_strict": [self.get_obj_pos, self.get_rel_rot]
            + [self.get_tactile_obs],
            "intrinsic_full_no_vel": [
                self.get_ft_pos,
                self.get_dof_pos,
                self.get_obj_pos,
                self.get_obj_rot,
                self.get_goal_pos,
                self.get_goal_rot,
                self.get_rel_rot,
            ]
            + [self.get_tactile_obs],
            "intrinsic_full_no_vel_strict": [
                self.get_dof_pos,
                self.get_obj_pos,
                self.get_obj_rot,
                self.get_goal_pos,
                self.get_goal_rot,
                self.get_rel_rot,
            ]
            + [self.get_tactile_obs],
            "intrinsic_full_no_vel_strict_no_proprio": [
                self.get_obj_pos,
                self.get_obj_rot,
                self.get_goal_pos,
                self.get_goal_rot,
                self.get_rel_rot,
            ]
            + [self.get_tactile_obs],
            "intrinsic_full": [
                self.get_ft_pos,
                self.get_dof_pos,
                self.get_obj_pos,
                self.get_obj_rot,
                self.get_goal_pos,
                self.get_goal_rot,
                self.get_rel_rot,
            ]
            + [
                self.get_dof_vel,
                self.get_obj_lin_vel,
                self.get_obj_ang_vel,
                self.get_ft_rot,
                self.get_ft_vel,
            ]
            + [self.get_tactile_obs],
            "intrinsic_full_strict": [
                self.get_dof_pos,
                self.get_obj_pos,
                self.get_obj_rot,
                self.get_goal_pos,
                self.get_goal_rot,
                self.get_rel_rot,
            ]
            + [
                self.get_dof_vel,
                self.get_obj_lin_vel,
                self.get_obj_ang_vel,
            ]
            + [self.get_tactile_obs],
            "intrinsic_full_strict_no_proprio": [
                self.get_obj_pos,
                self.get_obj_rot,
                self.get_goal_pos,
                self.get_goal_rot,
                self.get_rel_rot,
            ]
            + [
                self.get_obj_lin_vel,
                self.get_obj_ang_vel,
            ]
            + [self.get_tactile_obs],
            "intrinsic_full_state": [
                self.get_ft_pos,
                self.get_dof_pos,
                self.get_obj_pos,
                self.get_obj_rot,
                self.get_goal_pos,
                self.get_goal_rot,
                self.get_rel_rot,
            ]
            + [
                self.get_dof_vel,
                self.get_obj_lin_vel,
                self.get_obj_ang_vel,
                self.get_ft_rot,
                self.get_ft_vel,
            ]
            + [self.get_torque_obs]
            + [self.get_tactile_obs],
            "intrinsic_full_state_strict": [
                self.get_dof_pos,
                self.get_obj_pos,
                self.get_obj_rot,
                self.get_goal_pos,
                self.get_goal_rot,
                self.get_rel_rot,
            ]
            + [
                self.get_dof_vel,
                self.get_obj_lin_vel,
                self.get_obj_ang_vel,
            ]
            + [self.get_torque_obs]
            + [self.get_tactile_obs],
            "intrinsic_full_state_strict_no_proprio": [
                self.get_obj_pos,
                self.get_obj_rot,
                self.get_goal_pos,
                self.get_goal_rot,
                self.get_rel_rot,
            ]
            + [
                self.get_obj_lin_vel,
                self.get_obj_ang_vel,
            ]
            + [self.get_tactile_obs],
        }

        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 187,
            "intrinsic_openai": 42 + 5,
            "intrinsic_openai_strict": 42 + 5 - 15,
            "intrinsic_full_no_vel": 77 + 5,
            "intrinsic_full_no_vel_strict": 77 + 5 - 15,
            "intrinsic_full_no_vel_strict_no_proprio": 77 + 5 - 15 - 24,
            "intrinsic_full": 157 + 5,
            "intrinsic_full_strict": 157 + 5 - 15 - 20 - 30,
            "intrinsic_full_strict_no_proprio": 157 + 5 - 15 - 20 - 30 - 24 - 24,
            "intrinsic_full_state": 187 + 5,
            "intrinsic_full_state_strict": 187 + 5 - 15 - 20 - 30,
            "intrinsic_full_state_strict_no_proprio": 187
            + 5
            - 15
            - 20
            - 30
            - 24
            - 24
            - 30,
        }

        self.asymmetric_obs = self._task_cfg["env"]["asymmetric_observations"]
        # self.use_vel_obs = False

        self.fingertip_obs = True
        self.fingertips = [
            "robot0:ffdistal",
            "robot0:mfdistal",
            "robot0:rfdistal",
            "robot0:lfdistal",
            "robot0:thdistal",
        ]
        self.num_fingertips = len(self.fingertips)

        self.object_scale = torch.tensor([1.0, 1.0, 1.0])
        self.force_torque_obs_scale = 10.0

        num_states = 0
        if self.asymmetric_obs:
            num_states = 187

        self._num_observations = self.num_obs_dict[self.obs_type]
        self._num_actions = 20
        self._num_states = num_states

        self._control_mode = self._task_cfg["env"]["control_mode"]
        print("selected control mode: ", self._control_mode)

        InHandManipulationBaseTask.__init__(self, name=name, env=env)
        return

    def get_observations(self):
        self.get_object_goal_observations()

        self.fingertip_pos, self.fingertip_rot = self._hands._fingers.get_world_poses(
            clone=False
        )
        self.fingertip_pos -= self._env_pos.repeat((1, self.num_fingertips)).reshape(
            self.num_envs * self.num_fingertips, 3
        )
        self.fingertip_velocities = self._hands._fingers.get_velocities(clone=False)

        self.hand_dof_pos = self._hands.get_joint_positions(clone=False)
        self.hand_dof_vel = self._hands.get_joint_velocities(clone=False)

        if (
            self.obs_type
            in [
                "full_state",
                "intrinsic_full_state",
                "intrinsic_full_state_strict",
                "intrinsic_full_state_strict_no_proprio",
            ]
            or self.asymmetric_obs
        ):
            self.vec_sensor_tensor = (
                self._hands._physics_view.get_force_sensor_forces().reshape(
                    self.num_envs, 6 * self.num_fingertips
                )
            )

        self.obs_buf_offset = 0
        for obs_getter in self.obs_dict[self.obs_type]:
            obs = obs_getter()
            self.obs_buf[:, self.obs_buf_offset : len(obs)] = obs
            self.obs_buf_offset += len(obs)

        if self.asymmetric_obs:
            self.get_asymmetric_obs(
                True
            )  # may be modified to match openai's implementation in the future...

        observations = {self._hands.name: {"obs_buf": self.obs_buf}}
        return observations

    def get_ft_pos(self):
        return self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips)

    def get_ft_rot(self):
        return self.fingertip_rot.reshape(self.num_envs, 4 * self.num_fingertips)

    def get_ft_vel(self):
        return self.fingertip_velocities.reshape(self.num_envs, 6 * self.num_fingertips)

    def get_dof_pos(self):
        return unscale(
            self.hand_dof_pos,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        )

    def get_dof_vel(self):
        return self.vel_obs_scale * self.hand_dof_vel

    def get_obj_pos(self):
        return self.object_pos

    def get_obj_rot(self):
        return self.object_rot

    def get_obj_lin_vel(self):
        return self.object_linvel

    def get_obj_ang_vel(self):
        return self.vel_obs_scale * self.object_angvel

    def get_goal_pos(self):
        return self.goal_pos

    def get_goal_rot(self):
        return self.goal_rot

    def get_rel_rot(self):
        return quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

    def get_torque_obs(self):
        return self.force_torque_obs_scale * self.vec_sensor_tensor

    def get_tactile_obs(self):
        """Gets tacile/contact-related data."""

        # net contact forces
        net_contact_vec = self._shadow_hands._fingers.get_net_contact_forces(
            clone=False
        )
        net_contact_vec *= self.contact_obs_scale

        net_contact_val = torch.norm(
            net_contact_vec.view(self._num_envs, len(self.fingertips), 3), dim=-1
        )

        return (
            net_contact_vec.reshape(self.num_envs, 3 * self.num_fingertips),
            net_contact_val,
        )

    def get_asymmetric_obs(self):
        self.states_buf[:, 0 : self.num_hand_dofs] = self.get_dof_pos()
        self.states_buf[
            :, self.num_hand_dofs : 2 * self.num_hand_dofs
        ] = self.get_dof_vel()
        # self.states_buf[:, 2*self.num_hand_dofs:3*self.num_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

        obj_obs_start = 2 * self.num_hand_dofs  # 48
        self.states_buf[:, obj_obs_start : obj_obs_start + 3] = self.get_obj_pos()
        self.states_buf[
            :, obj_obs_start + 3 : obj_obs_start + 7
        ] = self.get_object_rot()
        self.states_buf[
            :, obj_obs_start + 7 : obj_obs_start + 10
        ] = self.get_obj_linvel()
        self.states_buf[
            :, obj_obs_start + 10 : obj_obs_start + 13
        ] = self.get_obj_ang_vel()

        goal_obs_start = obj_obs_start + 13  # 61
        self.states_buf[:, goal_obs_start : goal_obs_start + 3] = self.get_goal_pos()
        self.states_buf[
            :, goal_obs_start + 3 : goal_obs_start + 7
        ] = self.get_goal_rot()
        self.states_buf[
            :, goal_obs_start + 7 : goal_obs_start + 11
        ] = self.get_rel_rot()

        # fingertip observations, state(pose and vel) + force-torque sensors
        num_ft_states = 13 * self.num_fingertips  # 65
        num_ft_force_torques = 6 * self.num_fingertips  # 30

        fingertip_obs_start = goal_obs_start + 11  # 72
        self.states_buf[
            :, fingertip_obs_start : fingertip_obs_start + 3 * self.num_fingertips
        ] = self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips)
        self.states_buf[
            :,
            fingertip_obs_start
            + 3 * self.num_fingertips : fingertip_obs_start
            + 7 * self.num_fingertips,
        ] = self.fingertip_rot.reshape(self.num_envs, 4 * self.num_fingertips)
        self.states_buf[
            :,
            fingertip_obs_start
            + 7 * self.num_fingertips : fingertip_obs_start
            + 13 * self.num_fingertips,
        ] = self.fingertip_velocities.reshape(self.num_envs, 6 * self.num_fingertips)

        self.states_buf[
            :,
            fingertip_obs_start
            + num_ft_states : fingertip_obs_start
            + num_ft_states
            + num_ft_force_torques,
        ] = (
            self.force_torque_obs_scale * self.vec_sensor_tensor
        )

        # obs_end = 72 + 65 + 30 = 167
        # obs_total = obs_end + num_actions = 187
        obs_end = fingertip_obs_start + num_ft_states + num_ft_force_torques
        self.states_buf[:, obs_end : obs_end + self.num_actions] = self.actions

    def get_hand(self):
        hand_start_translation = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        hand_start_orientation = torch.tensor(
            [0.0, 0.0, -0.70711, 0.70711], device=self.device
        )

        shadow_hand = ShadowHand(
            prim_path=self.default_zero_env_path + "/shadow_hand",
            name="shadow_hand",
            translation=hand_start_translation,
            orientation=hand_start_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "shadow_hand",
            get_prim_at_path(shadow_hand.prim_path),
            self._sim_config.parse_actor_config("shadow_hand"),
        )
        shadow_hand.set_shadow_hand_properties(
            stage=self._stage, shadow_hand_prim=shadow_hand.prim
        )
        shadow_hand.set_motor_control_mode(
            stage=self._stage, shadow_hand_path=shadow_hand.prim_path
        )

        pose_dy, pose_dz = -0.39, 0.10
        return hand_start_translation, pose_dy, pose_dz

    def get_hand_view(self, scene):
        hand_view = ShadowHandView(
            prim_paths_expr="/World/envs/.*/shadow_hand", name="shadow_hand_view"
        )
        scene.add(hand_view._fingers)
        return hand_view
