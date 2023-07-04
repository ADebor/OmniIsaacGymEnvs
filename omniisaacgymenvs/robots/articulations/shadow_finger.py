from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from pxr import PhysxSchema
import os


class ShadowFinger(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "shadow_finger",
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        self._name = name

        assets_root_path = (
            os.path.dirname(os.path.realpath(__file__)) + "/../.." + "/assets"
        )
        self._usd_path = assets_root_path + "/Robots/shadow_finger_instanceable.usd"

        self._position = (
            torch.tensor([0.0, 0.0, 0.5]) if translation is None else translation
        )
        self._orientation = (
            torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation
        )
        add_reference_to_stage(self._usd_path, prim_path)
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

    def set_shadow_finger_properties(self, stage, shadow_finger_prim):
        for link_prim in shadow_finger_prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)
                rb.GetRetainAccelerationsAttr().Set(True)

    def set_motor_control_mode(self, stage, shadow_finger_path):
        joints_config = {
            "robot0_WRJ1": {"stiffness": 0.0, "damping": 0.0, "max_force": 4.785},
            "robot0_WRJ0": {"stiffness": 0.0, "damping": 0.0, "max_force": 2.175},
            
            "robot0_MFJ3": {"stiffness": 0.0, "damping": 0.0, "max_force": 0.9},
            "robot0_MFJ2": {"stiffness": 0.0, "damping": 0.0, "max_force": 0.9},
            "robot0_MFJ1": {"stiffness": 0.0, "damping": 0.0, "max_force": 0.7245},
        }

        # position control config:
        # joints_config = {
        #     "robot0_MFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
        #     "robot0_MFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
        #     "robot0_MFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
        #                 }

        for joint_name, config in joints_config.items():
            set_drive(
                f"{self.prim_path}/joints/{joint_name}",
                "angular",
                "position",
                0.0,
                config["stiffness"] * np.pi / 180,
                config["damping"] * np.pi / 180,
                config["max_force"],
            )
