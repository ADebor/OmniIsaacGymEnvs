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
from omniisaacgymenvs.robots.articulations.shadow_hand import ShadowHand
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omniisaacgymenvs.robots.articulations.views.shadow_hand_view import ShadowHandView

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
            "robot0_ffdistal",
            "robot0_mfdistal",
            "robot0_rfdistal",
            "robot0_lfdistal",
            "robot0_thdistal",
        ]
        self.num_fingertips = len(self.fingertips)

        # set number of observations and actions
        self._num_object_observations = 13  # (pos:3, rot:4, linvel:3, angvel:3)
        self._num_force_direction_obs = 15
        self._num_force_val_obs = 5
        self._num_tactile_observations = (
            self._num_force_direction_obs + self._num_force_val_obs
        )  # 20
        self._num_dof_observations = (
            3 * 24
        )  # (pos:1 + vel:1 + eff:1) * num_joints:24 (can be retrieved from hands rather than hardcoded)
        self._num_fingertip_observations = (
            65  # (pos:3, rot:4, linvel:3, angvel:3) * num_fingers:5
        )
        self._num_observations = (
            self._num_tactile_observations
            + self._num_object_observations
            + self._num_dof_observations
            + self._num_fingertip_observations
        )
        self._num_actions = 20

        # get cloning params - number of envs and spacing
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        # get metrics scaling factors and parameters
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.action_regul_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.fall_dist = self._task_cfg["env"]["fallDistance"]
        self.fall_penalty = self._task_cfg["env"]["fallPenalty"]
        self.no_fall_bonus = self._task_cfg["env"]["noFallBonus"]
        self.z_pos_scale = self._task_cfg["env"]["zPosScale"]
        self.contact_obs_scale = self._task_cfg["env"]["contactObsScale"]

        # get observation scaling factors
        self.vel_obs_scale = self._task_cfg["env"]["velObsScale"]

        # get object reset params
        self.reset_obj_pos_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_obj_rot_noise = self._task_cfg["env"]["resetRotationNoise"]

        # get shadow hand reset noise params
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
        self.x_unit_tensor = torch.tensor(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        self.av_factor = torch.tensor(
            self.av_factor, dtype=torch.float, device=self.device
        )

        self.fingertip_prim_path = self.default_zero_env_path + "/shadow_hand/"

        # live plotting init
        self._tactile_obs_visu_is_on = self._task_cfg["tactile_obs_visu"]
        if self._tactile_obs_visu_is_on:
            style.use("dark_background")
            self.env0_tactile_fig = plt.figure()
            self.env0_tactile_ax = self.env0_tactile_fig.add_subplot(111)
            self.env0_tactile_ax.set_ylabel("Contact force value [N] - scale: x{}".format(self.contact_obs_scale))
            self.env0_tactile_ax.set_ylim(bottom=0.0, top=1.5)
            self.env0_tactile_ax.tick_params(axis="x", labelrotation=45)
            self.env0_tactile_fig.suptitle("env0 - Hand-related observations")

        # handicap init 
        self._handicap_is_on = self._task_cfg["handicap"]["isOn"]
        if self._handicap_is_on:
            self._handicap_cfg = {
                "type": self._task_cfg["handicap"]["type"],
                "effort_scales": self._task_cfg["handicap"]["effort_scales"],
                "fingers": self._task_cfg["handicap"]["fingers"],
                "strict": self._task_cfg["handicap"]["strict"],
                "p_handic": self._task_cfg["handicap"]["p_handic"],
                "p_handic_release": self._task_cfg["handicap"]["p_handic_release"],
                # "delays": self._task_cfg["handicap"]["delays"],
                "effort_scale_gauss": self._task_cfg["handicap"]["effort_scale_gauss"],
                # "delay_scale_gauss": self._task_cfg["handicap"]["delay_scale_gauss"],
            }
            
            self.finger_control_handicap_init()

            # map dict of actuated dof indices and finger id (from FF to TH)
            self.actuated_dof_finger_map_dict = {
            0: [2, 7, 12],                  # FF
                1: [3, 8, 13],              # MF
                2: [4, 9, 14],              # RF
                3: [5, 10, 15, 20-3],       # LF NB: -3 - 17,18,19 not actuated
                4: [6, 11, 16, 21-3, 23-4], # TH NB: -4 - 22 not actuated either
            }

        self.dummy_bool = True


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
        hand_start_translation, pose_dy, pose_dz = self.get_hand()

        # get manipulated object
        self.get_object(hand_start_translation, pose_dy, pose_dz)

        # clones envs
        replicate_physics = False if self._dr_randomizer.randomize else True
        super().set_up_scene(scene, replicate_physics)

        # get a view of the cloned shadow hands and add it to the scene
        self._shadow_hands = self.get_hand_view(scene)
        scene.add(self._shadow_hands)

        # get fingers names
        if self._tactile_obs_visu_is_on:
            self.finger_names = self._shadow_hands._fingers.prim_paths
            self.finger_names = self.finger_names[:5]
            self.finger_names = [finger_name[30:] for finger_name in self.finger_names]
            bar_colors = ["tab:red", "tab:cyan", "tab:pink", "tab:olive", "tab:purple"]
            self.env0_tactile_bars = self.env0_tactile_ax.bar(
                self.finger_names, [0.0, 0.0, 0.0, 0.0, 0.0], color=bar_colors
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
        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/object/object",
            name="object_view",
            reset_xform_properties=False,
            # masses=torch.tensor([0.07087] * self._num_envs, device=self.device),
            masses=torch.tensor([0.700] * self._num_envs, device=self.device),
            # scales=torch.tensor(
            #     2 * torch.ones((self._num_envs, 3)), device=self.device
            # ),
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
        self._shadow_hands.switch_control_mode(mode="effort")

        # set effort mode
        self._shadow_hands.set_effort_modes(mode="force")

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

        # set defaut joint effort state
        self.hand_dof_default_eff = torch.zeros(
            self.num_hand_dofs, dtype=torch.float, device=self.device
        )

        # get manipulated objects' initial position and orientation (for reset), and set objects' initial velocities
        self.object_init_pos, self.object_init_rot = self._objects.get_world_poses()
        self.object_init_pos -= self._env_pos
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
            min_t=-self.hand_dof_effort_limits,
            max_t=self.hand_dof_effort_limits,
        )

        # apply handicap
        if self._handicap_is_on:
            self.apply_handicap()

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

    def get_observations(self) -> dict:
        """
        Implements logic to retrieve observation states.

        Returns:
            dict: Observations dictionary.
        """
        self.obs_buf_offset = 0
        self.get_object_observations()
        self.get_hand_observations()

        observations = {self._shadow_hands.name: {"obs_buf": self.obs_buf}}

        if self._tactile_obs_visu_is_on:
            bar_vals = self.obs_buf[0][-self._num_force_val_obs :].cpu().numpy()
            for i, bar_val in enumerate(bar_vals):
                self.env0_tactile_bars[i].set_height(bar_val)
            self.env0_tactile_fig.canvas.draw()
            self.env0_tactile_fig.canvas.flush_events()

        return observations

    def get_object_observations(self):
        """Gets manipulated object-related observations."""
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
        """Gets hand-related observations."""
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
        """Gets proprioceptive dof-related observations."""
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
            # should be a replicate of the sent actions if force-controlled
            self.force_torque_obs_scale
            * self.hand_dof_eff  # TODO: problem: gives nothing but 0.0's (seems to be a known bug, need to wait for next release)
        )

        self.obs_buf_offset += 3 * self.num_hand_dofs

    def get_tactile_observations(self):
        """Gets tacile/contact-related data."""

        # net contact forces
        net_contact_vec = self._shadow_hands._fingers.get_net_contact_forces(
            clone=False
        )
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
        self.rew_buf[:], self.reset_buf[:] = compute_hand_reward(
            self.reset_buf,
            self.object_pos,
            self.dist_reward_scale,
            self.fall_dist,
            self.object_init_pos,
            self.actions,
            self.action_regul_scale,
            self.fall_penalty,
            self.no_fall_bonus,
            self.z_pos_scale,
        )

    def is_done(self) -> None:
        """
        Implement logic to update dones/reset buffer.
        """
        pass

    def get_hand(self):
        """Creates ShadowHand instance and set initial pose."""
        # set Shadow hand initial position and orientation
        hand_start_translation = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        hand_start_orientation = torch.tensor(
            [0.0, 0.0, -0.70711, 0.70711], device=self.device
        )

        # custom usd file path
        # self._hand_usd_path = "../robots/usda/instanceable/tests/colortips/shadow_hand_instanceable.usd"
        # self._hand_usd_path = "/home/adebor/isaacsim_ws/OmniIsaacGymEnvs/omniisaacgymenvs/robots/usda/instanceable/tests/colortips/shadow_hand_instanceable.usd"
        # self._hand_usd_path = "/home/adebor/Documents/shadow_hand_instanceable.usd"
        self._hand_usd_path = None

        # create ShadowHand object and set it at initial pose
        self.shadow_hand = ShadowHand(
            prim_path=self.default_zero_env_path + "/shadow_hand",
            name="shadow_hand",
            translation=hand_start_translation,
            orientation=hand_start_orientation,
            usd_path=self._hand_usd_path,
        )

        # apply articulation settings to Shadow hand
        self._sim_config.apply_articulation_settings(
            "shadow_hand",
            get_prim_at_path(self.shadow_hand.prim_path),
            self._sim_config.parse_actor_config("shadow_hand"),
        )

        # set Shadow hand properties
        self.shadow_hand.set_shadow_hand_properties(
            stage=self._stage, shadow_hand_prim=self.shadow_hand.prim
        )

        # set motor control mode for the Shadow hand TODO: I don't know how this works exactly - set position target control...
        self.shadow_hand.set_motor_control_mode(
            stage=self._stage, shadow_hand_path=self.shadow_hand.prim_path
        )

        # set offset of the object to be manipulated (TODO: why here?)
        pose_dy, pose_dz = -0.39, 0.10

        return hand_start_translation, pose_dy, pose_dz

    def get_hand_view(self, scene):
        """Creates view of the cloned hands.

        Args:
            scene (Scene): Scene to add the view.

        Returns:
            ArticulationView: Cloned hands view.
        """
        # create a view of the Shadow hand
        hand_view = ShadowHandView(
            prim_paths_expr="/World/envs/.*/shadow_hand", name="shadow_hand_view"
        )

        # add the view of the fingers to the scene
        scene.add(hand_view._fingers)

        return hand_view

    def get_object(self, hand_start_translation, pose_dy, pose_dz):
        """Creates manipulated object prim.

        Args:
            hand_start_translation (torch.Tensor): Hand starting translation.
            pose_dy (float): Object starting offset along y-axis.
            pose_dz (float): Object starting offset along z-axis.
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

    def finger_control_handicap_init(
            self,
        # self, actions, type="partial", scales=[], fingers=[], n_fingers=None, durations=[], finger_perc=0.2, scale_mu=0.0, scale_sigma=1.0,
    ):
        
        if self._handicap_cfg["type"] not in ["partial", "delayed"]:
            logging.warn("Unvalid handicap type. Must be in [partial, total, delayed].")
            # self._handicap = False
            return

        if self._handicap_cfg["type"] == "partial":
            self.apply_handicap = self.apply_partial_handicap
        else:
            self.apply_handicap = self.apply_delayed_handicap

        self._handicap_effort_scale_gauss = self._handicap_cfg["effort_scale_gauss"]
        if self._handicap_effort_scale_gauss is not None:
            self._handicap_effort_scales = None
            self._handicap_effort_scale_gauss = np.asarray(self._handicap_effort_scale_gauss)
        else:
            self._handicap_effort_scales = self._handicap_cfg["effort_scales"]

        # self._handicap_fingers = self._handicap_cfg["fingers"]
        # self._handicap_num_fingers = len(self._handicap_fingers) if type(self._handicap_fingers) is list else self._handicap_fingers
        
        self._handicap_finger_p = np.asarray(self._handicap_cfg["p_handic"])
        self._handicap_finger_p_release = np.asarray(self._handicap_cfg["p_handic_release"])

        # create a view (no added memory)
        self._handicap_finger_p_torch = torch.from_numpy(self._handicap_finger_p).float().expand(self.num_envs,-1).to(self._device)  
        self._handicap_finger_p_release_torch = torch.from_numpy(self._handicap_finger_p_release).float().expand(self.num_envs,-1).to(self._device) 
        
        # self._handicap_strict = self._handicap_cfg["strict"]

        self.handicap_grid = np.zeros((self.num_envs, self.num_fingertips))
        self.handicap_grid_torch = torch.from_numpy(self.handicap_grid).to(self._device)
    
    def apply_partial_handicap(self):
        #TODO: not implemented: finger selection, strict/relax selection, delayed handicap

        # Bernoulli impairment pick
        # random_grid = np.random.binomial(1, self._handicap_finger_p, (self.num_envs, self.num_fingertips))
        random_grid_torch = torch.bernoulli(self._handicap_finger_p_torch).to(self._device)

        # Bernoulli impairment release pick
        # random_release_grid = np.random.binomial(1, self._handicap_finger_p_release, (self.num_envs, self.num_fingertips))
        random_release_grid_torch = torch.bernoulli(self._handicap_finger_p_release_torch).to(self._device)

        # release AND state (effectively released bits)
        # eff_release_indices = np.nonzero(np.logical_and(random_release_grid, self.handicap_grid))
        eff_release_indices_torch = torch.nonzero(torch.logical_and(random_release_grid_torch, self.handicap_grid_torch)).to(self._device)

        # previous impairment state OR new impairment state (to be impaired bits)
        # self.handicap_grid = np.logical_or(self.handicap_grid, random_grid)
        self.handicap_grid_torch = torch.logical_or(self.handicap_grid_torch, random_grid_torch)

        # effectively impaired bits (to be - effectively released)
        # self.handicap_grid[eff_release_indices] = 0
        self.handicap_grid_torch[eff_release_indices_torch] = 0

        # compute scaling factors from distribution
        if self._handicap_effort_scale_gauss:
            self._handicap_effort_scales = np.random.normal(self._handicap_effort_scale_gauss[:, 0], self._handicap_effort_scale_gauss[:, 0])
            
        # create scaling factor grid
        # scale_grid = np.ones((self.num_envs, len(self.actuated_dof_indices)))
        # for idx, scale in enumerate(self._handicap_effort_scales):
        #     scale_grid[:, self.actuated_dof_finger_map_dict[idx]] *= scale

        scale_grid_torch = torch.ones((self.num_envs, len(self.actuated_dof_indices))).to(self._device)
        for idx, scale in enumerate(self._handicap_effort_scales):
            scale_grid_torch[:, self.actuated_dof_finger_map_dict[idx]] *= scale

        # scale_grid = torch.from_numpy(scale_grid).to(self.device)

        # scale action tensor
        # self.efforts *= scale_grid
        self.efforts *= scale_grid_torch
    
    # def apply_delayed_handicap(self):
    #         print("delayed")
    #         return


# TorchScript functions


@torch.jit.script
def compute_hand_reward(
    reset_buf,
    object_pos,
    dist_reward_scale: float,
    fall_dist: float,
    object_init_pos,
    actions,
    action_regul_scale: float,
    fall_penalty: float,
    no_fall_bonus: float,
    z_pos_scale: float,
):
    """Computes task rewards."""

    # compute object distance to initial position
    dist_to_init = torch.norm(object_pos - object_init_pos, p=2, dim=-1)

    # reset condition
    resets = torch.where(dist_to_init >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    # distance to initial position reward
    dist_to_init_rew = dist_to_init * dist_reward_scale

    # action regularization
    action_regul = torch.sum(actions ** 2, dim=-1) * action_regul_scale

    # high position reward
    z_pos = object_pos[:, 2]
    high_pos_rew = z_pos * z_pos_scale

    # composite reward
    reward = dist_to_init_rew + action_regul + high_pos_rew

    # fall penalty - reward + penalty if too far, otherwise reward + no fall bonus
    reward = torch.where(dist_to_init >= fall_dist, reward + fall_penalty, reward + no_fall_bonus)


    return reward, resets


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """Randomizes rotation."""
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )
