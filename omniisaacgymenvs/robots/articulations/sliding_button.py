from typing import Optional

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

import torch
import os


class SlidingButton(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "sliding_button",
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]"""
        self._name = name

        assets_root_path = (
            os.path.dirname(os.path.realpath(__file__)) + "/../.." + "/assets"
        )
        self._usd_path = assets_root_path + "/Objects/sliding_button_instanceable.usd"
        add_reference_to_stage(self._usd_path, prim_path)

        self._position = (
            torch.tensor([0.0, 0.0, 0.4]) if translation is None else translation
        )
        self._orientation = (
            torch.tensor([0.1, 0.0, 0.0, 0.0]) if orientation is None else orientation
        )

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )
