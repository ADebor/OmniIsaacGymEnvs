from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class SlidingButtonView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "SlidingButtonView",
    ) -> None:
        """[summary]"""

        super().__init__(
            prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False
        )

        self._cylinders = RigidPrimView(
            prim_paths_expr="/World/envs/.*/button/button",
            name="cylinder_view",
            reset_xform_properties=False,
        )
