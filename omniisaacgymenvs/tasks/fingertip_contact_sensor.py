import omni 
import numpy as np

class FingertipContactSensor:
    def __init__(
        self,
        cs,
        prim_path,
        translation,
        radius=-1,
        color=(0.9, 0.2, 0.2, 5.0),
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
            translation=self._translation,
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


