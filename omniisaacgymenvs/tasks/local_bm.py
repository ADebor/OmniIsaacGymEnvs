"""
1-finger task to be determined
probably a task involving pressing a button whose internal resistance/stifness is dynamically changed across learning episodes, until a given threshold or sth
"""

# NVIDIA Omniverse imports
import omni.replicator.isaac as dr

# from omni.isaac.sensor import _sensor
# from .fingertip_contact_sensor import FingertipContactSensor
from omni.isaac.core.utils.torch import *
from omni.isaac.gym.vec_env import VecEnvBase
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omniisaacgymenvs.robots.articulations.shadow_finger import ShadowFinger
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage

# from omniisaacgymenvs.robots.articulations.views.shadow_hand_view import ShadowHandView
from omniisaacgymenvs.robots.articulations.views.shadow_finger_view import (
    ShadowFingerView,
)

# general imports
import torch
from gym import spaces
import numpy as np

# live plotting imports
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib

matplotlib.use("Qt5Agg")
plt.ion()
import logging


class LocalBenchmarkTask(
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
        # self.object_type = self._task_cfg["env"]["objectType"]
        # assert self.object_type in ["block"]

        # set object and force obs scale factors
        # self.object_scale = torch.tensor([1.0, 1.0, 1.0])
        self.force_torque_obs_scale = 10

        # get fingertips info
        self.fingertips = [
            "robot0_ffdistal",
            # "robot0_mfdistal",
            # "robot0_rfdistal",
            # "robot0_lfdistal",
            # "robot0_thdistal",
        ]
        self.num_fingertips = len(self.fingertips)

        # set number of observations and actions
        # self._num_object_observations = 13  # (pos:3, rot:4, linvel:3, angvel:3)
        self._num_object_observations = 1  # relative vertical position ?
        self._num_force_direction_obs = 15 / 5
        self._num_force_val_obs = 5 / 5
        self._num_tactile_observations = (
            self._num_force_direction_obs + self._num_force_val_obs
        )  # 20 / 5
        self._num_dof_observations = (
            3 * 4  # or 3 * 2 if phalange not considered?
        )  # (pos:1 + vel:1 + eff:1) * num_joints:24 (can be retrieved from hands rather than hardcoded)
        self._num_fingertip_observations = (
            65 / 5  # (pos:3, rot:4, linvel:3, angvel:3) * num_fingers:1
        )
        self._num_observations = (
            self._num_tactile_observations
            + self._num_object_observations
            + self._num_dof_observations
            + self._num_fingertip_observations
        )
        self._num_actions = 3  # or 1 if phalange not considered?

        # get cloning params - number of envs and spacing
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        # get metrics scaling factors and parameters
        # self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.action_regul_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.fall_dist = self._task_cfg["env"]["fallDistance"]
        self.throw_dist = self._task_cfg["env"]["throwDistance"]

        # self.fall_penalty = self._task_cfg["env"]["fallPenalty"]
        # self.no_fall_bonus = self._task_cfg["env"]["noFallBonus"]
        # self.z_pos_scale = self._task_cfg["env"]["zPosScale"]
        self.contact_obs_scale = self._task_cfg["env"]["contactObsScale"]

        # get observation scaling factors
        self.vel_obs_scale = self._task_cfg["env"]["velObsScale"]

        # get object reset params
        # TODO add here button position/high-rew zone location noise?
        self.reset_button_pos_noise = self._task_cfg["env"]["resetButtonPositionNoise"]
        self.reset_high_reward_zone_pos_noise = self._task_cfg["env"][
            "resetHighRewardZonePosNoise"
        ]
        # self.reset_obj_pos_noise = self._task_cfg["env"]["resetPositionNoise"]
        # self.reset_obj_rot_noise = self._task_cfg["env"]["resetRotationNoise"]

        # get shadow hand finger reset noise params
        self.reset_dof_pos_noise = self._task_cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self._task_cfg["env"]["resetDofVelRandomInterval"]
        self.reset_dof_eff_noise = self._task_cfg["env"]["resetDofEffRandomInterval"]

        self.av_factor = self._task_cfg["env"].get("averFactor", 0.1)

        # define custom action space
        self.action_space = spaces.Box(
            np.ones(self._num_actions) * -5.0, np.ones(self._num_actions) * 5.0
        )  # -10 and 10 could be -Inf Inf

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

        # set unit tensors for randomization of object position (in the x-y plane)
        # TODO add here tensors for randomization of button position/high-reward zone location?
        self.z_unit_tensor = torch.tensor(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        # self.x_unit_tensor = torch.tensor(
        #     [1, 0, 0], dtype=torch.float, device=self.device
        # ).repeat((self.num_envs, 1))
        # self.y_unit_tensor = torch.tensor(
        #     [0, 1, 0], dtype=torch.float, device=self.device
        # ).repeat((self.num_envs, 1))

        self.av_factor = torch.tensor(
            self.av_factor, dtype=torch.float, device=self.device
        )

        self.fingertip_prim_path = self.default_zero_env_path + "/shadow_finger/"

        # live plotting init
        self._tactile_obs_visu_is_on = self._task_cfg["tactile_obs_visu"]
        if self._tactile_obs_visu_is_on:
            style.use("dark_background")
            self.env0_tactile_fig = plt.figure()
            self.env0_tactile_ax = self.env0_tactile_fig.add_subplot(111)
            self.env0_tactile_ax.set_ylabel(
                "Contact force value [N] - scale: x{}".format(self.contact_obs_scale)
            )
            self.env0_tactile_ax.set_ylim(bottom=0.0, top=1.5)
            self.env0_tactile_ax.tick_params(axis="x", labelrotation=45)
            self.env0_tactile_fig.suptitle("env0 - Hand-related observations")

        # # handicap init
        # self._handicap_is_on = self._task_cfg["handicap"]["isOn"]
        # if self._handicap_is_on:
        #     self._handicap_cfg = {
        #         "type": self._task_cfg["handicap"]["type"],
        #         "effort_scales": self._task_cfg["handicap"]["effort_scales"],
        #         "fingers": self._task_cfg["handicap"]["fingers"],
        #         "strict": self._task_cfg["handicap"]["strict"],
        #         "p_handic": self._task_cfg["handicap"]["p_handic"],
        #         "p_handic_release": self._task_cfg["handicap"]["p_handic_release"],
        #         # "delays": self._task_cfg["handicap"]["delays"],
        #         "effort_scale_gauss": self._task_cfg["handicap"]["effort_scale_gauss"],
        #         # "delay_scale_gauss": self._task_cfg["handicap"]["delay_scale_gauss"],
        #     }

        #     self.finger_control_handicap_init()

        #     # map dict of actuated dof indices and finger id (from FF to TH)
        #     self.actuated_dof_finger_map_dict = {
        #         0: [2, 7, 12],  # FF
        #         1: [3, 8, 13],  # MF
        #         2: [4, 9, 14],  # RF
        #         3: [5, 10, 15, 20 - 3],  # LF NB: -3 - 17,18,19 not actuated
        #         4: [6, 11, 16, 21 - 3, 23 - 4],  # TH NB: -4 - 22 not actuated either
        #     }

    def set_up_scene(self, scene, replicate_physics=True) -> None:
        """
        Implements environment setup.

        Args:
            scene (Scene): Scene to add robot view.
            replicate_physics (bool, optional): Bool to clone physics using PhysX API for better performance. Defaults to True.
        """

        # self._cs = _sensor.acquire_contact_sensor_interface()

        # get USD stage, assets path and initialization params
        self._stage = get_current_stage()
        self._assets_root_path = get_assets_root_path()

        # get shadow hand Robot
        # hand_start_translation, pose_dy, pose_dz = self.get_hand()
        # get shadow hand finger Robot
        finger_start_translation, pose_dz = self.get_finger()

        # get manipulated object
        # self.get_object(hand_start_translation, pose_dy, pose_dz)
        self.get_object(finger_start_translation, pose_dz)

        # clones envs
        replicate_physics = False if self._dr_randomizer.randomize else True
        super().set_up_scene(scene, replicate_physics)

        # get a view of the cloned shadow fingers and add it to the scene
        self._shadow_fingers = self.get_finger_view()
        scene.add(self._shadow_fingers)

        # get fingers names
        # if self._tactile_obs_visu_is_on:
        #     self.finger_names = self._shadow_hands._fingers.prim_paths
        #     self.finger_names = self.finger_names[:5]
        #     self.finger_names = [finger_name[30:] for finger_name in self.finger_names]
        #     bar_colors = ["tab:red", "tab:cyan", "tab:pink", "tab:olive", "tab:purple"]
        #     self.env0_tactile_bars = self.env0_tactile_ax.bar(
        #         self.finger_names, [0.0, 0.0, 0.0, 0.0, 0.0], color=bar_colors
        #     )

        if self._tactile_obs_visu_is_on:
            self.env0_tactile_bars = self.env0_tactile_ax.bar(
                self.finger_names, [0.0], color=["tab:red"]
            )

        # create contact sensors
        # self._contact_sensors = {}
        # self.env_name_offset = self.default_zero_env_path.find("env_")
        # self._contact_sensor_translation = Gf.Vec3d(0.0, 0.0, 0.026)

        # for prim_path in self._shadow_hands.prim_paths:
        #     env_name = prim_path[
        #         self.env_name_offset : self.env_name_offset + len("env_") + 1
        #     ]
        #     self._contact_sensors[env_name] = {}
        #     for finger_name in self.fingertips:
        #         fingertip_path = prim_path + "/" + finger_name
        #         self._contact_sensors[env_name][finger_name] = FingertipContactSensor(
        #             self._cs,
        #             fingertip_path,
        #             radius=0.01,
        #             translation=self._contact_sensor_translation,
        #         )

        # create a view of the cloned objects and add it to the scene
        self._buttons = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/button/button",
            name="button_view",
            reset_xform_properties=False,
            # masses=torch.tensor([0.07087] * self._num_envs, device=self.device),
            masses=torch.tensor([0.700] * self._num_envs, device=self.device),
            # scales=torch.tensor(
            #     2 * torch.ones((self._num_envs, 3)), device=self.device
            # ),
        )
        scene.add(self._butons)

        # apply domain randomization if needed
        if self._dr_randomizer.randomize:
            self._dr_randomizer.apply_on_startup_domain_randomization(self)

    def post_reset(self):
        """
        Implements any logic required for simulation on-start.
        """
        # get number of dof and actuated dof indices
        self.num_finger_dofs = self._shadow_fingers.num_dof
        self.actuated_dof_indices = self._shadow_fingers.actuated_dof_indices

        # switch control mode (ensure effort mode is selected)
        self._shadow_fingers.switch_control_mode(mode="effort")

        # set effort mode
        self._shadow_fingers.set_effort_modes(mode="force")

        # get dof position limits of the shadow finger
        dof_limits = self._shadow_fingers.get_dof_limits()
        self.finger_dof_pos_lower_limits, self.finger_dof_pos_upper_limits = torch.t(
            dof_limits[0].to(self.device)
        )

        # get maximum efforts for articulations in the view
        effort_limits = (
            self._shadow_fingers.get_max_efforts()
        )  # query all prims and all joints - return M(#prims)xK(#joints) tensor
        effort_limits = effort_limits[
            0
        ]  # gives 4 values - 4 joints of the Shadow finger, not 3 dofs
        self.finger_dof_effort_limits = effort_limits[
            self.actuated_dof_indices
        ]  # gives 3 values - 3 actuated joints

        # initialize dof default pos and vel and eff of the shadow finger
        self.finger_dof_default_pos = torch.zeros(
            self.num_finger_dofs, dtype=torch.float, device=self.device
        )
        self.finger_dof_default_vel = torch.zeros(
            self.num_finger_dofs, dtype=torch.float, device=self.device
        )
        # set defaut joint effort state
        self.finger_dof_default_eff = torch.zeros(
            self.num_finger_dofs, dtype=torch.float, device=self.device
        )

        # get manipulated objects' initial position and orientation (for reset), and set objects' initial velocities
        self.button_init_pos, self.button_init_rot = self._buttons.get_world_poses()
        self.button_init_pos -= self._env_pos
        self.button_init_velocities = torch.zeros_like(
            self._buttons.get_velocities(), dtype=torch.float, device=self.device
        )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Implements logic to be performed before physics steps.

        Args:
            actions (torch.Tensor): Actions to be taken.
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
            min_t=-self.finger_dof_effort_limits,
            max_t=self.finger_dof_effort_limits,
        )

        # set joint effort
        self._shadow_fingers.set_joint_efforts(
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

    def get_observations(self) -> dict:
        """
        Implements logic to retrieve observation states.

        Returns:
            dict: Observations dictionary.
        """
        self.obs_buf_offset = 0
        self.get_button_observations()
        self.get_finger_observations()

        observations = {self._shadow_fingers.name: {"obs_buf": self.obs_buf}}

        if self._tactile_obs_visu_is_on:
            bar_vals = self.obs_buf[0][-self._num_force_val_obs :].cpu().numpy()
            for i, bar_val in enumerate(bar_vals):
                self.env0_tactile_bars[i].set_height(bar_val)
            self.env0_tactile_fig.canvas.draw()
            self.env0_tactile_fig.canvas.flush_events()

        return observations

    # def get_object_observations(self):
    #     """Gets manipulated object-related observations."""
    #     # get data
    #     self.object_pos, self.object_rot = self._objects.get_world_poses(
    #         clone=False
    #     )  # NB: if clone then returns a clone of the internal buffer
    #     self.object_pos -= self._env_pos
    #     self.object_velocities = self._objects.get_velocities(clone=False)
    #     self.object_linvel = self.object_velocities[:, 0:3]
    #     self.object_angvel = self.object_velocities[:, 3:6]

    #     # populate observation buffer
    #     self.obs_buf[
    #         :, self.obs_buf_offset + 0 : self.obs_buf_offset + 3
    #     ] = self.object_pos
    #     self.obs_buf[
    #         :, self.obs_buf_offset + 3 : self.obs_buf_offset + 7
    #     ] = self.object_rot
    #     self.obs_buf[
    #         :, self.obs_buf_offset + 7 : self.obs_buf_offset + 10
    #     ] = self.object_linvel
    #     self.obs_buf[:, self.obs_buf_offset + 10 : self.obs_buf_offset + 13] = (
    #         self.object_angvel * self.vel_obs_scale
    #     )

    #     self.obs_buf_offset += 13

    def get_button_observations(self):
        """Gets manipulated button-related observations."""

        # get data
        self.button_pos, _ = self._buttons.get_world_poses(
            clone=False
        )  # NB: if clone then returns a clone of the internal buffer
        self.button_pos -= self._env_pos
        self.button_z_pos = self.button_pos[2]
        print("button z pos: ", self.button_z_pos)

        self.button_velocities = self._buttons.get_velocities(clone=False)
        self.button_linvel = self.button_velocities[:, 0:3]
        self.button_angvel = self.button_velocities[:, 3:6]
        print("button lin vel: ", self.button_linvel)
        print("button ang vel: ", self.button_angvel)

        # populate observation buffer
        self.obs_buf[
            :, self.obs_buf_offset + 0 : self.obs_buf_offset + 1
        ] = self.button_z_pos
        self.obs_buf[
            :, self.obs_buf_offset + 1 : self.obs_buf_offset + 4
        ] = self.button_linvel
        self.obs_buf[:, self.obs_buf_offset + 4 : self.obs_buf_offset + 7] = (
            self.button_angvel * self.vel_obs_scale
        )
        # TODO: add relative position of the button and the high reward zone

        self.obs_buf_offset += 7

    def get_finger_observations(self):
        """Gets finger-related observations."""
        # proprioceptive observations
        self.get_pc_observations()

        # sensing observations
        self.get_tactile_observations()

    def get_pc_observations(self):
        """Gets proprioceptive observations."""
        # fingertip observations
        self.get_fingertip_observations()

        # dof observations
        self.get_dof_observations()

    def get_fingertip_observations(self):
        """Gets proprioceptive fingertip-related observations."""
        # get data
        (
            self.fingertip_pos,
            self.fingertip_rot,
        ) = self._shadow_fingers._tips.get_world_poses(clone=False)

        self.fingertip_pos -= self._env_pos.repeat((1, self.num_fingertips)).reshape(
            self.num_envs * self.num_fingertips, 3
        )
        self.fingertip_vel = self._shadow_fingers._tips.get_velocities(clone=False)

        # populate observation buffer
        self.obs_buf[
            :, self.obs_buf_offset + 0 : self.obs_buf_offset + 3
        ] = self.fingertip_pos.reshape(
            self.num_envs, 3 * self.num_fingertips
        )  # 1*3 fingertip pos
        self.obs_buf[
            :, self.obs_buf_offset + 3 : self.obs_buf_offset + 7
        ] = self.fingertip_rot.reshape(
            self.num_envs, 4 * self.num_fingertips
        )  # 1*4 fingertip rot
        self.obs_buf[
            :, self.obs_buf_offset + 7 : self.obs_buf_offset + 13
        ] = self.fingertip_vel.reshape(
            self.num_envs, 6 * self.num_fingertips
        )  # 1*6 fingertip vel

        self.obs_buf_offset += 13

    def get_dof_observations(self):
        """Gets proprioceptive dof-related observations."""

        # get data
        self.finger_dof_pos = self._shadow_fingers.get_joint_positions(clone=False)
        self.finger_dof_vel = self._shadow_fingers.get_joint_velocities(clone=False)
        self.finger_dof_eff = self._shadow_fingers.get_applied_joint_efforts(
            clone=False
        )

        # populate observation buffer
        self.obs_buf[
            :, self.obs_buf_offset + 0 : self.obs_buf_offset + self.num_finger_dofs
        ] = unscale(
            self.finger_dof_pos,
            self.finger_dof_pos_lower_limits,
            self.finger_dof_pos_upper_limits,
        )
        self.obs_buf[
            :,
            self.obs_buf_offset
            + self.num_finger_dofs : self.obs_buf_offset
            + 2 * self.num_finger_dofs,
        ] = (
            self.vel_obs_scale * self.finger_dof_vel
        )
        self.obs_buf[
            :,
            self.obs_buf_offset
            + 2 * self.num_finger_dofs : self.obs_buf_offset
            + 3 * self.num_finger_dofs,
        ] = (
            # should be a replicate of the sent actions if force-controlled
            self.force_torque_obs_scale
            * self.hand_dof_eff  # TODO: problem: gives nothing but 0.0's (seems to be a known bug, need to wait for next release)
        )

        self.obs_buf_offset += 3 * self.num_finger_dofs

    def get_tactile_observations(self):
        """Gets tacile/contact-related data."""

        # net contact forces
        net_contact_vec = self._shadow_fingers._tips.get_net_contact_forces(clone=False)
        net_contact_vec *= self.contact_obs_scale

        net_contact_val = torch.norm(
            net_contact_vec.view(self._num_envs, len(self.fingertips), 3), dim=-1
        )

        self.obs_buf[
            :,
            self.obs_buf_offset
            + 0 : self.obs_buf_offset
            + self._num_force_direction_obs,
        ] = net_contact_vec.reshape(self.num_envs, 3 * self.num_fingertips)
        self.obs_buf[
            :,
            self.obs_buf_offset
            + self._num_force_direction_obs : self.obs_buf_offset
            + self._num_force_direction_obs
            + self._num_force_val_obs,
        ] = net_contact_val

        # detailed force sensors
        # for env, sensor_dict in self._contact_sensors.items():
        #     for finger, sensor in sensor_dict.items():
        #         (
        #             force_val,
        #             direction,
        #             impulses,
        #             dts,
        #             normals,
        #             positions,
        #             reading_ts,
        #             sim_ts,
        #         ) = sensor.get_data()

        #         print(
        #             "-- Fingertip sensor for finger {} in env {} --\n".format(
        #                 finger, env
        #             )
        #             + "force: {} \n ".format(force_val)
        #             + "impulses: {} \n ".format(impulses)
        #             + "direction: {} \n ".format(direction)
        #             + "from normals: {} \n ".format(normals)
        #             + "reading time: {} \n ".format(reading_ts)
        #             + "sim time: {} \n".format(sim_ts)
        #         )

        return

    def calculate_metrics(self) -> None:
        """
        Implements logic to compute rewards.
        """
        self.rew_buf[:], self.reset_buf[:] = compute_finger_reward(
            self.reset_buf,
            self.button_pos,
            # self.dist_reward_scale,
            self.fall_dist,
            self.throw_dist,
            self.button_init_pos,
            self.actions,
            self.action_regul_scale,
            # self.fall_penalty,
            # self.no_fall_bonus,
            # self.z_pos_scale,
        )

    def is_done(self) -> None:
        """
        Implement logic to update dones/reset buffer.
        """
        pass

    # def get_hand(self):
    #     """Creates ShadowHand instance and set initial pose."""
    #     # set Shadow hand initial position and orientation
    #     hand_start_translation = torch.tensor([0.0, 0.0, 0.5], device=self.device)
    #     hand_start_orientation = torch.tensor(
    #         [0.0, 0.0, -0.70711, 0.70711], device=self.device
    #     )

    #     # custom usd file path
    #     # self._hand_usd_path = "../robots/usda/instanceable/tests/colortips/shadow_hand_instanceable.usd"
    #     # self._hand_usd_path = "/home/adebor/isaacsim_ws/OmniIsaacGymEnvs/omniisaacgymenvs/robots/usda/instanceable/tests/colortips/shadow_hand_instanceable.usd"
    #     # self._hand_usd_path = "/home/adebor/Documents/shadow_hand_instanceable.usd"
    #     self._hand_usd_path = None

    #     # create ShadowHand object and set it at initial pose
    #     self.shadow_hand = ShadowHand(
    #         prim_path=self.default_zero_env_path + "/shadow_hand",
    #         name="shadow_hand",
    #         translation=hand_start_translation,
    #         orientation=hand_start_orientation,
    #         usd_path=self._hand_usd_path,
    #     )

    #     # apply articulation settings to Shadow hand
    #     self._sim_config.apply_articulation_settings(
    #         "shadow_hand",
    #         get_prim_at_path(self.shadow_hand.prim_path),
    #         self._sim_config.parse_actor_config("shadow_hand"),
    #     )

    #     # set Shadow hand properties
    #     self.shadow_hand.set_shadow_hand_properties(
    #         stage=self._stage, shadow_hand_prim=self.shadow_hand.prim
    #     )

    #     # set motor control mode for the Shadow hand TODO: I don't know how this works exactly - set position target control...
    #     self.shadow_hand.set_motor_control_mode(
    #         stage=self._stage, shadow_hand_path=self.shadow_hand.prim_path
    #     )

    #     # set offset of the object to be manipulated (TODO: why here?)
    #     pose_dy, pose_dz = -0.39, 0.10

    #     return hand_start_translation, pose_dy, pose_dz

    def get_finger(self):
        """Creates ShadowFinger instance and sets initial pose."""
        finger_start_translation = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        finger_start_orientation = torch.tensor(
            [0.0, 0.0, 0.0, 0.0], device=self.device
        )

        self._finger_usd_path = "../assets/Robots/shadow_finger.usd"

        # create ShadowFinger object and set it at initial pose
        self.shadow_finger = ShadowFinger(
            prim_path=self.default_zero_env_path + "/shadow_finger",
            name="shadow_finger",
            translation=finger_start_translation,
            orientation=finger_start_orientation,
            usd_path=self._fingre_usd_path,
        )

        # apply articulation settings to Shadow finger
        self._sim_config.apply_articulation_settings(
            "shadow_finger",
            get_prim_at_path(self.shadow_finger.prim_path),
            self._sim_config.parse_actor_config("shadow_finger"),
        )

        # set Shadow finger properties
        self.shadow_finger.set_shadow_finger_properties(
            stage=self._stage, shadow_finger_prim=self.shadow_finger.prim
        )

        # set motor control mode for the Shadow finger TODO: I don't know how this works exactly - set position target control...
        self.shadow_finger.set_motor_control_mode(
            stage=self._stage, shadow_finger_path=self.shadow_finger.prim_path
        )

        # set offset of the object to be manipulated (TODO: why here?)
        # pose_dy, pose_dz = -0.39, 0.10
        pose_dz = 0.1

        return finger_start_translation, pose_dz

    # def get_hand_view(self, scene):
    #     """Creates view of the cloned hands.

    #     Args:
    #         scene (Scene): Scene to add the view.

    #     Returns:
    #         ArticulationView: Cloned hands view.
    #     """
    #     # create a view of the Shadow hand
    #     hand_view = ShadowHandView(
    #         prim_paths_expr="/World/envs/.*/shadow_hand", name="shadow_hand_view"
    #     )

    #     # add the view of the fingers to the scene
    #     scene.add(hand_view._fingers)

    #     return hand_view

    def get_finger_view(self):
        """Creates a view of the cloned fingers"""
        finger_view = ShadowFingerView(
            prim_path_expr="World/envs/.*/shadow_finger", name="shadow_finger_view"
        )

        return finger_view

    # def get_object(self, hand_start_translation, pose_dy, pose_dz):
    #     """Creates manipulated object prim.

    #     Args:
    #         hand_start_translation (torch.Tensor): Hand starting translation.
    #         pose_dy (float): Object starting offset along y-axis.
    #         pose_dz (float): Object starting offset along z-axis.
    #     """
    #     # get hand translation and object offset to set object translation
    #     self.object_start_translation = hand_start_translation.clone()
    #     self.object_start_translation[1] += pose_dy
    #     self.object_start_translation[2] += pose_dz

    #     # set object orientation
    #     self.object_start_orientation = torch.tensor(
    #         [1.0, 0.0, 0.0, 0.0], device=self.device
    #     )

    #     # get object asset and add reference to stage
    #     self.object_usd_path = (
    #         f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
    #     )
    #     add_reference_to_stage(
    #         self.object_usd_path, self.default_zero_env_path + "/object"
    #     )

    #     # create object prim
    #     obj = XFormPrim(
    #         prim_path=self.default_zero_env_path + "/object/object",
    #         name="object",
    #         translation=self.object_start_translation,
    #         orientation=self.object_start_orientation,
    #         scale=self.object_scale,
    #     )
    #     self._sim_config.apply_articulation_settings(
    #         "object",
    #         get_prim_at_path(obj.prim_path),
    #         self._sim_config.parse_actor_config("object"),
    #     )
    def get_object(self, finger_start_translation, pose_dz):
        """Creates manipulated button"""
        self.object_start_translation = finger_start_translation.clone()
        self.object_start_translation[2] += pose_dz

        # set object orientation
        self.object_start_orientation = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )

        # get object asset and add reference to stage
        self.object_usd_path = "../assets/Objects/sliding_button.usd"
        add_reference_to_stage(
            self.object_usd_path, self.default_zero_env_path + "/button"
        )

        # create object prim
        obj = XFormPrim(
            prim_path=self.default_zero_env_path + "/button/button",
            name="button",
            translation=self.object_start_translation,
            orientation=self.object_start_orientation,
            scale=self.object_scale,
        )

        self._sim_config.apply_articulation_settings(
            "button",
            get_prim_at_path(obj.prim_path),
            self._sim_config.parse_actor_config("button"),
        )

    def reset_idx(self, env_ids):
        """Resets environments (Shadow hands and manipulated objects) specified as argument.

        Args:
            env_ids (torch.Tensor): IDs of the cloned environments to be reset.
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

        self.reset_buf[env_ids] = 0


# TorchScript functions


@torch.jit.script
def compute_finger_reward(
    reset_buf,
    reset_goal_buf,
    button_pos,
    target_pos,
    consecutive_successes,
    fall_dist: float,
    fall_penalty: float,
    button_init_pos,
    actions,
    action_regul_scale: float,
    success_tolerance: float,
    success_reward: float,
    failure_reward: float,
    max_consecutive_successes: int,
    max_episode_length: float,
    av_factor: float,
):
    """Computes task rewards and resets."""

    ###############
    #   Rewards   #
    ###############

    # goal reward
    goal_dist = torch.norm(button_pos - target_pos, p=2, dim=-1)

    dist_rew = torch.where(
        goal_dist <= success_tolerance,
        success_reward * torch.ones_like(reset_goal_buf),
        failure_reward * torch.ones_like(reset_goal_buf),
    )

    # action regularization
    action_regul = torch.sum(actions**2, dim=-1) * action_regul_scale

    # composite reward
    reward = dist_rew + action_regul

    # fall penalty: distance to the goal is larger than a threshold #TODO: maybe consider "throw" penalty (opposite direction)
    dist_to_init = torch.norm(button_pos - button_init_pos, p=2, dim=-1)
    reward = torch.where(dist_to_init >= fall_dist, reward + fall_penalty, reward)

    ##############
    #   Resets   #
    ##############

    ## goal zone resets ##

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(
        goal_dist <= success_tolerance,
        torch.ones_like(reset_goal_buf),
        reset_goal_buf,
    )
    successes = successes + goal_resets

    ## button resets ##

    # check fall condition
    resets = torch.where(
        dist_to_init >= fall_dist, torch.ones_like(reset_buf), reset_buf
    )

    # check max success condition
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(
            goal_dist <= success_tolerance,
            torch.zeros_like(progress_buf),
            progress_buf,
        )
        resets = torch.where(
            successes >= max_consecutive_successes, torch.ones_like(resets), resets
        )

    # max episode length resets
    resets = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets
    )

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(
            progress_buf >= max_episode_length - 1, reward + 0.5 * fall_penalty, reward
        )

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets
        + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """Randomizes rotation."""
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )