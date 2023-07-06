from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

import torch


class ShadowFingerView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "ShadowFingerView",
        track_contact_forces=True,
        prepare_contact_sensors=True,
    ) -> None:
        super().__init__(
            prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False
        )

        # define fingertip view to track contact forces
        self._tips = RigidPrimView(
            prim_paths_expr="/World/envs/.*/shadow_finger/robot0.*distal",
            name="finger_view",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )

    @property
    def actuated_dof_indices(self):
        return self._actuated_dof_indices

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
        self.actuated_joint_names = [
            # "robot0_MFJ3",
            "robot0_MFJ2",
            "robot0_MFJ1",
        ]
        self._actuated_dof_indices = list()

        for joint_name in self.actuated_joint_names:
            self._actuated_dof_indices.append(self.get_dof_index(joint_name))
        self._actuated_dof_indices.sort()

        limit_stiffness = torch.tensor(
            [30.0] * self.num_fixed_tendons, device=self._device
        )
        damping = torch.tensor([0.1] * self.num_fixed_tendons, device=self._device)
        self.set_fixed_tendon_properties(
            dampings=damping, limit_stiffnesses=limit_stiffness
        )
