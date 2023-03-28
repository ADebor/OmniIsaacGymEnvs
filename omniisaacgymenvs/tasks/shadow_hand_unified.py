import omni.replicator.isaac as dr
from omni.isaac.core.utils.torch import *
from omni.isaac.gym.vec_env import VecEnvBase
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omniisaacgymenvs.robots.articulations.shadow_hand import ShadowHand
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omniisaacgymenvs.robots.articulations.views.shadow_hand_view import ShadowHandView

import torch
import wandb
from gym import spaces
import numpy as np


class ShadowHandCustomTask(
    RLTask  # RLTask contains rl_games-specific config parameters and buffers
):
    """
    Shadow Hand task where main task logic is implemented. Inherits from RLTask.
    """

    def __init__(
        self,
        name: str,  # name of the task
        sim_config: SimConfig,  # SimConfig instance for parsing cfg
        env: VecEnvBase,  # env instance of VecEnvBase or inherited class
        offset=None,  # transform offset in World
    ) -> None:
        # parse configurations
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # assert manipulated object type
        self.object_type = self._task_cfg["env"]["objectType"]
        assert self.object_type in ["block"]

        # set object and force obs scale factors
        self.object_scale = torch.tensor([1.0, 1.0, 1.0])
        self.force_torque_obs_scale = 10

        # get fingertips info
        self.fingertips = [
            "robot0:ffdistal",
            "robot0:mfdistal",
            "robot0:rfdistal",
            "robot0:lfdistal",
            "robot0:thdistal",
        ]
        self.num_fingertips = len(self.fingertips)

        # set number of observations and actions
        self._num_object_observations = 13  # 5*(pos:3, rot:4, linvel:3, angvel:3)
        # self._num_tactile_observations = 5                    # 5 tactile sensors
        self._num_dof_observations = (
            3 * 25
        )  # (pos:1 + vel:1 + eff:1) * num_joints:25 (can be retrieved from hands rather than hardcoded)
        self._num_fingertip_observations = (
            65  # (pos:3, rot:4, linvel:3, angvel:3) * num_fingers:5
        )
        self._num_observations = (
            # self._tactile_observations +\
            self._num_object_observations
            + self._num_dof_observations
            + self._num_fingertip_observations
        )
        self._num_actions = 20

        # get cloning params - number of envs and spacing
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        # get metrics params TODO adapt to needs
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        # self.success_tolerance = self._task_cfg["env"]["successTolerance"]
        # self.reach_goal_bonus = self._task_cfg["env"]["reachGoalBonus"]
        self.fall_dist = self._task_cfg["env"]["fallDistance"]
        self.fall_penalty = self._task_cfg["env"]["fallPenalty"]
        self.rot_eps = self._task_cfg["env"]["rotEps"]
        self.vel_obs_scale = self._task_cfg["env"]["velObsScale"]

        # get object reset params
        self.reset_obj_pos_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_obj_rot_noise = self._task_cfg["env"]["resetRotationNoise"]

        # get shadow hand reset params
        self.reset_dof_pos_noise = self._task_cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self._task_cfg["env"]["resetDofVelRandomInterval"]
        self.reset_dof_eff_noise = self._task_cfg["env"]["resetDofEffRandomInterval"]

        # get shadow hand control settings
        self.hand_dof_speed_scale = self._task_cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self._task_cfg["env"]["useRelativeControl"]
        self.act_moving_average = self._task_cfg["env"]["actionsMovingAverage"]

        # get end of episode settings
        self.max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.reset_time = self._task_cfg["env"].get("resetTime", -1.0)
        # self.print_success_stat = self._task_cfg["env"]["printNumSuccesses"]
        # self.max_consecutive_successes = self._task_cfg["env"][
        #     "maxConsecutiveSuccesses"
        # ]
        self.av_factor = self._task_cfg["env"].get("averFactor", 0.1)

        self.total_successes = 0
        self.total_resets = 0
        self.dt = 1.0 / 60
        control_freq_inv = self._task_cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(
                round(self.reset_time / (control_freq_inv * self.dt))
            )
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)
        
        # define custom action space
        self.action_space = spaces.Box(np.ones(self._num_actions) * -10.0, np.ones(self._num_actions) * 10.0)

        # ---------------------------- #

        # call parent class's __init__
        super().__init__(name, env)  # will get device

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.consecutive_successes = torch.zeros(
            1, dtype=torch.float, device=self.device
        )
        self.randomization_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # set unit tensors for randomization of object position
        self.x_unit_tensor = torch.tensor(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        # self.z_unit_tensor = torch.tensor(
        #     [0, 0, 1], dtype=torch.float, device=self.device
        # ).repeat((self.num_envs, 1))

        self.av_factor = torch.tensor(
            self.av_factor, dtype=torch.float, device=self.device
        )

    def set_up_scene(self, scene, replicate_physics=True) -> None:
        """
        Implements environment setup.

        Args:
            scene (Scene): Scene to add robot view.
            replicate_physics (bool, optional): _description_. Defaults to True.
        """
        # get USD stage, assets path and initialization params
        self._stage = get_current_stage()
        self._assets_root_path = get_assets_root_path()
        hand_start_translation, pose_dy, pose_dz = self.get_hand()
        self.get_object(hand_start_translation, pose_dy, pose_dz)

        replicate_physics = False if self._dr_randomizer.randomize else True
        super().set_up_scene(
            scene, replicate_physics
        )  # clones envs - replicate_physics: clone physics using PhysX API for better performance

        # get a view of the shadow hands and add it to the scene
        self._shadow_hands = self.get_hand_view(scene)
        scene.add(self._shadow_hands)

        # create a view of the objects and add it to the scene
        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/object/object",
            name="object_view",
            reset_xform_properties=False,
            masses=torch.tensor([0.07087] * self._num_envs, device=self.device),
        )
        scene.add(self._objects)

        # apply domain randomization if needed
        if self._dr_randomizer.randomize:
            self._dr_randomizer.apply_on_startup_domain_randomization(self)

    def post_reset(self):
        """
        Implements any logic required for simulation on-start.
        """
        # get number of dof and actuated dof indices
        self.num_hand_dofs = self._shadow_hands.num_dof
        self.actuated_dof_indices = self._shadow_hands.actuated_dof_indices

        # switch control mode (ensure effort mode is selected)
        # self._shadow_hands.switch_control_mode(mode="position")
        # self._shadow_hands.switch_control_mode(mode="velocity")
        self._shadow_hands.switch_control_mode(mode="effort")

        # set effort mode
        # self._shadow_hands.set_effort_modes(mode="acceleration")
        self._shadow_hands.set_effort_modes(mode="force")

        # check effort mode
        print("Shadow hand effort modes: ", self._shadow_hands.get_effort_modes())
        
        # get dof position limits of the shadow hand
        dof_limits = self._shadow_hands.get_dof_limits()
        self.hand_dof_pos_lower_limits, self.hand_dof_pos_upper_limits = torch.t(
            dof_limits[0].to(self.device)
        )

        # get maximum efforts for articulations in the view
        effort_limits = (
            self._shadow_hands.get_max_efforts()
        )  # query all prims and all joints - return M(#prims)xK(#joints) tensor
        effort_limits = effort_limits[
            0
        ]  # gives 24 values - 24 joints of the Shadow hand, not 20 dofs
        self.hand_dof_effort_limits = effort_limits[
            self.actuated_dof_indices
        ]  # gives 20 values - 20 actuated joints

        # initialize dof default pos and vel of the shadow hand
        self.hand_dof_default_pos = torch.zeros(
            self.num_hand_dofs, dtype=torch.float, device=self.device
        )
        self.hand_dof_default_vel = torch.zeros(
            self.num_hand_dofs, dtype=torch.float, device=self.device
        )

        # set defaut joint effort state TODO check if correct
        # default_efforts = torch.zeros((1, self.num_hand_dofs), dtype=torch.float, device=self.device)
        # self._shadow_hands.set_joints_default_state(positions=None, velocities=None, efforts=default_efforts)
        self.hand_dof_default_eff = torch.zeros(
            self.num_hand_dofs, dtype=torch.float, device=self.device
        )

        # get manipulated objects' initial position and orientation (for reset), and set objects' initial velocities
        self.object_init_pos, self.object_init_rot = self._objects.get_world_poses()
        self.object_init_pos -= (
            self._env_pos
        )  # TODO: check if correct: correction of objects' pos related to cloning
        self.object_init_velocities = torch.zeros_like(
            self._objects.get_velocities(), dtype=torch.float, device=self.device
        )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Implements logic to be performed before physics steps.

        Args:
            actions (torch.Tensor): _description_
        """
        # check that the simulator is playing
        if not self._env._world.is_playing():
            return

        reset_buf = self.reset_buf.clone()

        # get indices of envs that need to be reset and reset them
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # get actions to device
        self.actions = actions.clone().to(self.device)

        # clamp actions using effort limits
        self.efforts = tensor_clamp(
            self.actions,
            min_t=torch.zeros_like(self.hand_dof_effort_limits, dtype=torch.float32),
            max_t=self.hand_dof_effort_limits,
        )

        # set joint effort
        self._shadow_hands.set_joint_efforts(
            efforts=self.efforts,
            indices=None,  # all prims in the view
            joint_indices=self.actuated_dof_indices,
        )

        if self._dr_randomizer.randomize:
            rand_envs = torch.where(
                self.randomization_buf >= self._dr_randomizer.min_frequency,
                torch.ones_like(self.randomization_buf),
                torch.zeros_like(self.randomization_buf),
            )
            rand_env_ids = torch.nonzero(torch.logical_and(rand_envs, reset_buf))
            dr.physics_view.step_randomization(rand_env_ids)
            self.randomization_buf[rand_env_ids] = 0

        print("Efforts/Actions: \n {} \n".format(self.efforts))
        # # wandb logging
        # wandb.log(
        #     {
        #         "action_0": self.efforts[0],
        #         "action_1": self.efforts[1],
        #         "action_2": self.efforts[2],
        #         "action_3": self.efforts[3],
        #         "action_4": self.efforts[4],
        #     }
        # )

    def get_observations(self) -> dict:
        """
        Implements logic to retrieve observation states.

        Returns:
            dict: _description_
        """
        self.obs_buf_offset = 0
        self.get_object_observations()
        self.get_hand_observations()

        observations = {self._shadow_hands.name: {"obs_buf": self.obs_buf}}

        return observations

    def get_object_observations(self):
        """_summary_"""
        # get data
        self.object_pos, self.object_rot = self._objects.get_world_poses(
            clone=False
        )  # NB: if clone then returns a clone of the internal buffer
        self.object_pos -= self._env_pos
        self.object_velocities = self._objects.get_velocities(clone=False)
        self.object_linvel = self.object_velocities[:, 0:3]
        self.object_angvel = self.object_velocities[:, 3:6]

        # populate observation buffer
        self.obs_buf[
            :, self.obs_buf_offset + 0 : self.obs_buf_offset + 3
        ] = self.object_pos
        self.obs_buf[
            :, self.obs_buf_offset + 3 : self.obs_buf_offset + 7
        ] = self.object_rot
        self.obs_buf[
            :, self.obs_buf_offset + 7 : self.obs_buf_offset + 10
        ] = self.object_linvel
        self.obs_buf[:, self.obs_buf_offset + 10 : self.obs_buf_offset + 13] = (
            self.object_angvel * self.vel_obs_scale
        )

        self.obs_buf_offset += 13

    def get_hand_observations(self):
        """_summary_"""
        # proprioceptive observations
        self.get_pc_observations()

        # sensing observations
        self.get_tactile_observations()

    def get_pc_observations(self):
        # fingertip observations
        self.get_fingertip_observations()

        # dof observations
        self.get_dof_observations()

    def get_fingertip_observations(self):
        """_summary_"""
        # get data
        (
            self.fingertip_pos,
            self.fingertip_rot,
        ) = self._shadow_hands._fingers.get_world_poses(clone=False)
        self.fingertip_pos -= self._env_pos.repeat((1, self.num_fingertips)).reshape(
            self.num_envs * self.num_fingertips, 3
        )
        self.fingertip_vel = self._shadow_hands._fingers.get_velocities(clone=False)

        # populate observation buffer
        self.obs_buf[
            :, self.obs_buf_offset + 0 : self.obs_buf_offset + 15
        ] = self.fingertip_pos.reshape(
            self.num_envs, 3 * self.num_fingertips
        )  # 5*3 fingertip pos
        self.obs_buf[
            :, self.obs_buf_offset + 15 : self.obs_buf_offset + 35
        ] = self.fingertip_rot.reshape(
            self.num_envs, 4 * self.num_fingertips
        )  # 5*4 fingertip rot
        self.obs_buf[
            :, self.obs_buf_offset + 35 : self.obs_buf_offset + 65
        ] = self.fingertip_vel.reshape(
            self.num_envs, 6 * self.num_fingertips
        )  # 5*6 fingertip vel

        self.obs_buf_offset += 65

    def get_dof_observations(self):
        # get data
        self.hand_dof_pos = self._shadow_hands.get_joint_positions(clone=False)
        self.hand_dof_vel = self._shadow_hands.get_joint_velocities(clone=False)
        self.hand_dof_eff = self._shadow_hands.get_applied_joint_efforts(clone=False)

        # populate observation buffer
        self.obs_buf[
            :, self.obs_buf_offset + 0 : self.obs_buf_offset + self.num_hand_dofs
        ] = unscale(
            self.hand_dof_pos,
            self.hand_dof_pos_lower_limits,
            self.hand_dof_pos_upper_limits,
        )
        self.obs_buf[
            :,
            self.obs_buf_offset
            + self.num_hand_dofs : self.obs_buf_offset
            + 2 * self.num_hand_dofs,
        ] = (
            self.vel_obs_scale * self.hand_dof_vel
        )
        self.obs_buf[
            :,
            self.obs_buf_offset
            + 2 * self.num_hand_dofs : self.obs_buf_offset
            + 3 * self.num_hand_dofs,
        ] = (
            # should be replicate of the sent actions if force-controlled
            self.force_torque_obs_scale
            * self.hand_dof_eff  # TODO: problem: gives nothing but 0.0's (seems to be a known bug, need to wait for next release)
        )

        self.obs_buf_offset += 3 * self.num_hand_dofs
        print(
            "observation buffer: \n {} \nof size: {}.".format(
                self.obs_buf, self.obs_buf.size()
            )
        )

    def get_tactile_observations(self):
        pass  # TODO

    def calculate_metrics(self) -> None:
        """
        Implements logic to compute rewards.
        """
        # self.rew_buf = self.compute_rewards()
        self.rew_buf[:], self.reset_buf[:] = compute_hand_reward(
            self.reset_buf,
            self.object_pos,
            self.dist_reward_scale,
            self.fall_dist,
            self.object_init_pos,
        )

    def is_done(self) -> None:
        """
        Implement logic to update dones/reset buffer.
        """
        # self.reset_buf = self.compute_resets()
        pass

    def get_hand(self):
        # set Shadow hand initial position and orientation
        hand_start_translation = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        hand_start_orientation = torch.tensor(
            [0.0, 0.0, -0.70711, 0.70711], device=self.device
        )

        # create ShadowHand object and set it at initial pose
        shadow_hand = ShadowHand(
            prim_path=self.default_zero_env_path + "/shadow_hand",
            name="shadow_hand",
            translation=hand_start_translation,
            orientation=hand_start_orientation,
        )

        # apply articulation settings to Shadow hand
        self._sim_config.apply_articulation_settings(
            "shadow_hand",
            get_prim_at_path(shadow_hand.prim_path),
            self._sim_config.parse_actor_config("shadow_hand"),
        )

        # set Shadow hand properties
        shadow_hand.set_shadow_hand_properties(
            stage=self._stage, shadow_hand_prim=shadow_hand.prim
        )

        # set motor control mode for the Shadow hand TODO: I don't know how this works exactly - set position target control...
        shadow_hand.set_motor_control_mode(
            stage=self._stage, shadow_hand_path=shadow_hand.prim_path
        )

        # set offset of the object to be manipulated (TODO: why here?)
        pose_dy, pose_dz = -0.39, 0.10

        return hand_start_translation, pose_dy, pose_dz

    def get_hand_view(self, scene):
        # create a view of the Shadow hand
        hand_view = ShadowHandView(
            prim_paths_expr="/World/envs/.*/shadow_hand", name="shadow_hand_view"
        )

        # add the view's fingers to the scene
        scene.add(hand_view._fingers)

        return hand_view

    def get_object(self, hand_start_translation, pose_dy, pose_dz):
        """_summary_

        Args:
            hand_start_translation (_type_): _description_
            pose_dy (_type_): _description_
            pose_dz (_type_): _description_
        """
        # get hand translation and object offset to set object translation
        self.object_start_translation = hand_start_translation.clone()
        self.object_start_translation[1] += pose_dy
        self.object_start_translation[2] += pose_dz

        # set object orientation
        self.object_start_orientation = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )

        # get object asset and add reference to stage
        self.object_usd_path = (
            f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        )
        add_reference_to_stage(
            self.object_usd_path, self.default_zero_env_path + "/object"
        )

        # create object prim
        obj = XFormPrim(
            prim_path=self.default_zero_env_path + "/object/object",
            name="object",
            translation=self.object_start_translation,
            orientation=self.object_start_orientation,
            scale=self.object_scale,
        )
        self._sim_config.apply_articulation_settings(
            "object",
            get_prim_at_path(obj.prim_path),
            self._sim_config.parse_actor_config("object"),
        )

    def reset_idx(self, env_ids):
        """Resets environments (Shadow hands and manipulated objects) specified as argument.

        Args:
            env_ids (_type_): _description_
        """
        indices = env_ids.to(dtype=torch.int32)

        # create noise - (obj_pos_x, obj_pos_y, obj_pos_z, obj_rot_x, obj_rot_y,
        #               hand_pos_1, ..., hand_pos_#dof,
        #               hand_vel_1, ..., hand_vel_#dof,
        #               )
        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), self.num_hand_dofs * 3 + 5), device=self.device
        )

        # (noisy) reset of manipulated object - pos, rot, and vel
        new_object_pos = (
            self.object_init_pos[env_ids]
            + self.reset_obj_pos_noise * rand_floats[:, 0:3]
            + self._env_pos[env_ids]
        )  # add noise to default pos
        new_object_rot = randomize_rotation(
            rand_floats[:, 3],
            rand_floats[:, 4],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )  # randomize rot
        self._objects.set_world_poses(new_object_pos, new_object_rot, indices)

        object_velocities = torch.zeros_like(
            self.object_init_velocities, dtype=torch.float, device=self.device
        )  # zero vel
        self._objects.set_velocities(object_velocities[env_ids], indices)

        # (noisy) reset of Shadow hand - pos, vel, and efforts
        delta_max = self.hand_dof_pos_upper_limits - self.hand_dof_default_pos
        delta_min = self.hand_dof_pos_lower_limits - self.hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (
            rand_floats[:, 5 : 5 + self.num_hand_dofs] + 1.0
        )

        pos = (
            self.hand_dof_default_pos + self.reset_dof_pos_noise * rand_delta
        )  # add noise to default pos
        dof_pos = torch.zeros((self.num_envs, self.num_hand_dofs), device=self.device)
        dof_pos[env_ids, :] = pos
        self._shadow_hands.set_joint_positions(dof_pos[env_ids], indices)

        vel = (
            self.hand_dof_default_vel
            + self.reset_dof_vel_noise
            * rand_floats[:, 5 + self.num_hand_dofs : 5 + self.num_hand_dofs * 2]
        )  # add noise to default vel
        dof_vel = torch.zeros((self.num_envs, self.num_hand_dofs), device=self.device)
        dof_vel[env_ids, :] = vel
        self._shadow_hands.set_joint_velocities(dof_vel[env_ids], indices)

        # control targets reset
        # self.prev_targets[env_ids, :self.num_hand_dofs] = pos
        # self.cur_targets[env_ids, :self.num_hand_dofs] = pos
        # self.hand_dof_targets[env_ids, :] = pos

        # self._shadow_hands.set_joint_position_targets(self.hand_dof_targets[env_ids], indices)
        # self._shadow_hands.set_joint_positions(dof_pos[env_ids], indices)
        # self._shadow_hands.set_joint_velocities(dof_vel[env_ids], indices)

        # TODO: reset hand's joint efforts - check if works
        eff = (
            self.hand_dof_default_eff
            + self.reset_dof_eff_noise  # 0.0
            * rand_floats[:, 5 + self.num_hand_dofs * 2 : 5 + self.num_hand_dofs * 3]
        )
        dof_eff = torch.zeros((self._num_envs, self.num_hand_dofs), device=self.device)
        dof_eff[env_ids, :] = eff
        self._shadow_hands.set_joint_efforts(
            efforts=dof_eff[env_ids], indices=indices, joint_indices=None
        )

        # self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        # self.successes[env_ids] = 0


# TorchScript functions


@torch.jit.script
def compute_hand_reward(
    # rew_buf,
    reset_buf,
    # reset_goal_buf,
    # progress_buf,
    # successes,
    # consecutive_successes,
    # max_episode_length: float,
    object_pos,
    # object_rot,
    # target_pos,
    # target_rot,
    dist_reward_scale: float,
    # rot_reward_scale: float,
    # rot_eps: float,
    # actions,
    # action_penalty_scale: float,
    # success_tolerance: float,
    # reach_goal_bonus: float,
    fall_dist: float,
    # fall_penalty: float,
    # max_consecutive_successes: int,
    # av_factor: float,
    object_init_pos,
):
    # goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    goal_dist = torch.norm(object_pos - object_init_pos, p=2, dim=-1)

    # Orientation alignment for the cube in hand and goal cube
    # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)) # changed quat convention

    # dist_rew = goal_dist * dist_reward_scale
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    # goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    # reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    # reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    # if max_consecutive_successes > 0:
    #     # Reset progress buffer on goal envs if max_consecutive_successes > 0
    #     progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
    #     resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    # resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    # if max_consecutive_successes > 0:
    # reward = torch.where(progress_buf >= max_episode_length - 1, reward + 0.5 * fall_penalty, reward)

    # num_resets = torch.sum(resets)
    # finished_cons_successes = torch.sum(successes * resets.float())

    # cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    reward = goal_dist * dist_reward_scale

    # return reward, resets, goal_resets, progress_buf, successes, cons_successes
    return reward, resets


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )
