# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

import carb
from pxr import Gf, PhysxSchema

import omni


class ShadowHand(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "shadow_hand",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
        fingertips=None,
        cs=None,
    ) -> None:
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._usd_path = (
                assets_root_path
                + "/Isaac/Robots/ShadowHand/shadow_hand_instanceable.usd"
            )

        self._position = (
            torch.tensor([0.0, 0.0, 0.5]) if translation is None else translation
        )
        self._orientation = (
            torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation
        )

        # create the contact sensors
        self.contact_sensors = {}
        for finger_name in fingertips:
            fingertip_path = (
                prim_path + "/" + finger_name
            ) 
            self.contact_sensors[finger_name] = FingertipContactSensor(
                cs, fingertip_path, radius=0.01, translation=self._position
            )

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

    def set_shadow_hand_properties(self, stage, shadow_hand_prim):
        for link_prim in shadow_hand_prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(True)
                rb.GetRetainAccelerationsAttr().Set(True)

    def set_motor_control_mode(self, stage, shadow_hand_path):
        joints_config = {
            "robot0_WRJ1": {"stiffness": 5, "damping": 0.5, "max_force": 4.785},
            "robot0_WRJ0": {"stiffness": 5, "damping": 0.5, "max_force": 2.175},
            "robot0_FFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_FFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_FFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "robot0_MFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_MFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_MFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "robot0_RFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_RFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_RFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "robot0_LFJ4": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_LFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_LFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_LFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "robot0_THJ4": {"stiffness": 1, "damping": 0.1, "max_force": 2.3722},
            "robot0_THJ3": {"stiffness": 1, "damping": 0.1, "max_force": 1.45},
            "robot0_THJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.99},
            "robot0_THJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.99},
            "robot0_THJ0": {"stiffness": 1, "damping": 0.1, "max_force": 0.81},
        }

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

class FingertipContactSensor:
    def __init__(
        self,
        cs,
        prim_path,
        translation,
        radius=-1,
        color=(1.0, 0.2, 0.1, 1.0),
        visualize=True,
    ):
        self._cs = cs
        self._prim_path = prim_path
        self._radius = radius
        self._color = color
        self._visualize = visualize
        self._translation = translation
        self.set_force_sensor()

    def set_force_sensor(
        self,
    ):
        """
        Create force sensor and attach on specified prim.

        Args:
            prim_path (str): Path of the prim on which to create the contact sensor.
            radius (int, optional): Radius of the contact sensor sphere. Defaults to -1.
        """

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateContactSensor",
            path="/contact_sensor",
            parent=self._prim_path,
            min_threshold=0,
            max_threshold=10000000,
            radius=self._radius,
            color=self._color,
            sensor_period=-1,
            translation=Gf.Vec3d(0.0, 0.0, 0.026),
            visualize=self._visualize,
        )
        self._sensor_path = self._prim_path + "/contact_sensor"

    def get_data(self):
        """Gets contact sensor (processed) data."""

        raw_data = self._cs.get_contact_sensor_raw_data(self._sensor_path)
        reading = self._cs.get_sensor_sim_reading(self._sensor_path)

        force_val = reading.value
        normals = np.array(
            [[x, y, z] for (x, y, z) in raw_data["normal"]]
        )  # global coordinates

        if reading.inContact:
            # get global force direction vector
            direction = np.sum(normals, axis=0)
            direction = direction / np.linalg.norm(direction)

        else:
            direction = [0, 0, 0]

        positions = raw_data["position"]  # global coordinates TODO compute local ones
        impulses = raw_data["impulse"]  # global coordinates
        dts = raw_data["dt"]
        reading_ts = reading.time  # TODO use timestamps for log
        sim_ts = raw_data["time"]

        return (
            force_val,
            direction,
            impulses,
            dts,
            normals,
            positions,
            reading_ts,
            sim_ts,
        )
