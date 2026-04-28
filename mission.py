#!/usr/bin/env python3

import colorsys
import math
import time
from typing import Dict, List, Optional, Tuple

import rospy
from clover import srv
from clover.srv import SetLEDEffect
from sensor_msgs.msg import Range
from std_srvs.srv import Trigger


class MissionAbort(Exception):
    """Raised when the mission should stop and land."""


def finite(*values: float) -> bool:
    """True only when every supplied value is a finite number."""
    return all(math.isfinite(v) for v in values)


class CloverComplexPatternMission:
    def __init__(self) -> None:
        rospy.init_node("clover_complex_sensor_led_pattern")

        self.pattern_radius = float(rospy.get_param("~pattern_radius", 0.75))
        self.takeoff_altitude = float(rospy.get_param("~takeoff_altitude", 1.00))
        self.max_altitude = float(rospy.get_param("~max_altitude", 1.35))
        self.speed = float(rospy.get_param("~speed", 0.35))
        self.arrival_tolerance = float(rospy.get_param("~arrival_tolerance", 0.18))
        self.max_wait_per_waypoint = float(rospy.get_param("~max_wait_per_waypoint", 12.0))
        self.low_cell_voltage = float(rospy.get_param("~low_cell_voltage", 3.55))
        self.obstacle_distance = float(rospy.get_param("~obstacle_distance", 0.70))

        # Latest Range readings, keyed by sensor name. Missing sensors are okay.
        self.ranges: Dict[str, float] = {}
        self._subscribe_range("down", str(rospy.get_param("~down_range_topic", "/rangefinder/range")))
        self._subscribe_range("front", str(rospy.get_param("~front_range_topic", "/rangefinder_front/range")))

        self._connect_clover_services()
        self.led_available = self._connect_led_service()

    def _subscribe_range(self, name: str, topic: str) -> None:
        if not topic:
            return
        rospy.Subscriber(topic, Range, lambda msg, n=name: self._range_callback(n, msg), queue_size=1)
        rospy.loginfo("Subscribed to %s rangefinder topic: %s", name, topic)

    def _range_callback(self, name: str, msg: Range) -> None:
        if msg.min_range <= msg.range <= msg.max_range and math.isfinite(msg.range):
            self.ranges[name] = msg.range

    def _connect_clover_services(self) -> None:
        needed = ["get_telemetry", "navigate", "set_position", "land"]
        for service_name in needed:
            rospy.loginfo("Waiting for Clover service: %s", service_name)
            rospy.wait_for_service(service_name, timeout=30)

        self.get_telemetry = rospy.ServiceProxy("get_telemetry", srv.GetTelemetry)
        self.navigate = rospy.ServiceProxy("navigate", srv.Navigate)
        self.set_position = rospy.ServiceProxy("set_position", srv.SetPosition)
        self.land = rospy.ServiceProxy("land", Trigger)

    def _connect_led_service(self) -> bool:
        try:
            rospy.wait_for_service("led/set_effect", timeout=2.0)
            self.set_led_effect = rospy.ServiceProxy("led/set_effect", SetLEDEffect)
            rospy.loginfo("LED service connected: led/set_effect")
            return True
        except rospy.ROSException:
            rospy.logwarn("LED service led/set_effect is not available; mission will fly without LEDs.")
            return False

    def led(self, effect: str, r: int, g: int, b: int,
            brightness: int = 180, duration: float = 0.0,
            priority: int = 0, base: bool = True) -> None:
        """Set LED effect, but never abort the flight just because LEDs fail."""
        if not self.led_available:
            return
        try:
            self.set_led_effect(
                effect=effect,
                r=max(0, min(255, int(r))),
                g=max(0, min(255, int(g))),
                b=max(0, min(255, int(b))),
                brightness=max(0, min(255, int(brightness))),
                duration=float(duration),
                priority=int(priority),
                base=bool(base),
            )
        except rospy.ServiceException as exc:
            rospy.logwarn("LED effect failed: %s", exc)

    def telemetry(self, frame_id: str = "map"):
        try:
            return self.get_telemetry(frame_id=frame_id)
        except rospy.ServiceException as exc:
            raise MissionAbort("Telemetry service failed: {}".format(exc))

    def choose_localization_frame(self) -> str:
        """
        Prefer ArUco map localization for indoor flights.
        Fall back to map, then body-relative movement if absolute localization is unavailable.
        """
        self.led("blink", 0, 80, 255, brightness=120, duration=0, priority=2, base=True)
        preferred_frames = ["aruco_map", "map"]

        for frame in preferred_frames:
            for _ in range(10):
                if rospy.is_shutdown():
                    raise MissionAbort("ROS shutdown requested")
                tel = self.telemetry(frame)
                if getattr(tel, "connected", False) and finite(tel.x, tel.y, tel.z):
                    rospy.loginfo("Using localization frame: %s", frame)
                    return frame
                rospy.sleep(0.2)

        rospy.logwarn("No finite aruco_map/map pose found. Falling back to body-relative pattern.")
        return "body"

    def safety_check(self, frame_id: str = "map", target_z: Optional[float] = None) -> None:
        """Abort on clear unsafe telemetry. Sensor readings may be unavailable and are handled gently."""
        tel = self.telemetry(frame_id if frame_id != "body" else "map")

        if hasattr(tel, "connected") and not tel.connected:
            raise MissionAbort("Flight controller connection lost")

        if finite(getattr(tel, "cell_voltage", float("nan"))) and tel.cell_voltage < self.low_cell_voltage:
            raise MissionAbort("Battery cell voltage is low: {:.2f} V".format(tel.cell_voltage))

        # Do not allow commands above the configured altitude ceiling.
        if target_z is not None and target_z > self.max_altitude:
            raise MissionAbort("Target altitude {:.2f} m exceeds max_altitude {:.2f} m".format(target_z, self.max_altitude))

        # Downward rangefinder is used only as a sanity check because floors may be uneven.
        down = self.ranges.get("down")
        if down is not None and down > self.max_altitude + 0.45:
            raise MissionAbort("Downward rangefinder reads too high: {:.2f} m".format(down))

    def front_obstacle_detected(self) -> bool:
        front = self.ranges.get("front")
        return front is not None and front < self.obstacle_distance

    def avoid_front_obstacle(self) -> None:
        """Pause, light orange, and back up slightly if an optional front rangefinder sees something."""
        rospy.logwarn("Front obstacle detected. Hovering and backing up slightly.")
        self.led("blink_fast", 255, 90, 0, brightness=180, duration=1.5, priority=10, base=False)
        self.set_position(frame_id="body")  # hover at the current position
        rospy.sleep(1.0)
        self.navigate(x=-0.35, y=0.0, z=0.0, yaw=float("nan"), speed=0.2, frame_id="body")
        rospy.sleep(2.0)

    def wait_until_arrived(self, x: float, y: float, z: float, frame_id: str) -> bool:
        """Wait for an absolute-frame waypoint. Returns False on timeout, not fatal."""
        start = time.time()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and time.time() - start < self.max_wait_per_waypoint:
            self.safety_check(frame_id=frame_id, target_z=z)

            if self.front_obstacle_detected():
                self.avoid_front_obstacle()

            tel = self.telemetry(frame_id)
            if finite(tel.x, tel.y, tel.z, tel.vx, tel.vy, tel.vz):
                distance = math.sqrt((tel.x - x) ** 2 + (tel.y - y) ** 2 + (tel.z - z) ** 2)
                speed = math.sqrt(tel.vx ** 2 + tel.vy ** 2 + tel.vz ** 2)
                if distance < self.arrival_tolerance and speed < 0.18:
                    return True
            rate.sleep()

        rospy.logwarn("Waypoint wait timed out; continuing mission.")
        return False

    def go_to_absolute(self, x: float, y: float, z: float, yaw: float,
                       frame_id: str, label: str = "") -> None:
        self.safety_check(frame_id=frame_id, target_z=z)
        rospy.loginfo("Waypoint %s: x=%.2f y=%.2f z=%.2f frame=%s", label, x, y, z, frame_id)
        self.navigate(x=x, y=y, z=z, yaw=yaw, speed=self.speed, frame_id=frame_id)
        self.wait_until_arrived(x, y, z, frame_id)

    def color_from_phase(self, phase: float) -> Tuple[int, int, int]:
        red, green, blue = colorsys.hsv_to_rgb(phase % 1.0, 0.95, 1.0)
        return int(red * 255), int(green * 255), int(blue * 255)

    def generate_clover_points(self, center_x: float, center_y: float, base_z: float) -> List[Tuple[float, float, float, float]]:
        """Generate a smooth four-leaf rose curve around the takeoff position."""
        points: List[Tuple[float, float, float, float]] = []
        count = 40
        altitude_wave = 0.16

        for i in range(count + 1):
            theta = 2.0 * math.pi * i / count
            radius = self.pattern_radius * math.sin(2.0 * theta)  # four-petal rose
            x = center_x + radius * math.cos(theta)
            y = center_y + radius * math.sin(theta)
            z = base_z + altitude_wave * math.sin(4.0 * theta)
            z = max(0.65, min(self.max_altitude, z))
            yaw = theta + math.pi / 2.0
            points.append((x, y, z, yaw))

        # Finish at the center for a tidy landing approach.
        points.append((center_x, center_y, base_z, float("nan")))
        return points

    def takeoff(self) -> None:
        self.led("blink", 30, 180, 255, brightness=160, duration=0, priority=5, base=True)
        rospy.loginfo("Taking off to %.2f m", self.takeoff_altitude)
        self.navigate(x=0.0, y=0.0, z=self.takeoff_altitude, yaw=float("nan"),
                      speed=0.35, frame_id="body", auto_arm=True)
        rospy.sleep(5.0)

    def fly_absolute_clover(self, frame_id: str) -> None:
        tel = self.telemetry(frame_id)
        if not finite(tel.x, tel.y, tel.z):
            raise MissionAbort("Localization became unavailable before absolute pattern")

        center_x, center_y = tel.x, tel.y
        base_z = max(0.75, min(self.max_altitude - 0.10, tel.z))
        points = self.generate_clover_points(center_x, center_y, base_z)

        self.led("rainbow", 0, 0, 0, brightness=120, duration=2.0, priority=4, base=False)
        rospy.sleep(0.5)

        for index, (x, y, z, yaw) in enumerate(points):
            r, g, b = self.color_from_phase(index / max(1, len(points) - 1))
            effect = "blink" if index % 8 == 0 else "fill"
            self.led(effect, r, g, b, brightness=150, duration=0.7 if effect == "blink" else 0, priority=3, base=True)
            self.go_to_absolute(x, y, z, yaw, frame_id, label="{:02d}".format(index))

    def fly_body_relative_backup_pattern(self) -> None:
        """
        Relative fallback if no absolute localization is available.
        It makes a small box/diamond/cross pattern using optical-flow/body-relative control.
        """
        rospy.loginfo("Flying body-relative fallback pattern")
        self.led("blink", 180, 0, 255, brightness=150, duration=1.5, priority=4, base=False)

        # dx, dy, dz, yaw degrees, LED hue phase
        moves = [
            (0.55, 0.00, 0.00, 0, 0.00),
            (0.00, 0.55, 0.10, 45, 0.12),
            (-0.55, 0.00, 0.00, 90, 0.25),
            (0.00, -0.55, -0.10, 135, 0.37),
            (0.40, 0.40, 0.08, 180, 0.50),
            (-0.80, 0.00, 0.00, -135, 0.62),
            (0.40, -0.40, -0.08, -90, 0.75),
            (0.00, 0.00, 0.00, 0, 0.90),
        ]

        for index, (dx, dy, dz, yaw_deg, phase) in enumerate(moves):
            self.safety_check(target_z=self.takeoff_altitude + dz)
            if self.front_obstacle_detected():
                self.avoid_front_obstacle()
            r, g, b = self.color_from_phase(phase)
            self.led("fill", r, g, b, brightness=140, duration=0, priority=3, base=True)
            rospy.loginfo("Relative segment %d: dx=%.2f dy=%.2f dz=%.2f", index, dx, dy, dz)
            self.navigate(x=dx, y=dy, z=dz, yaw=math.radians(yaw_deg),
                          speed=0.30, frame_id="body")
            rospy.sleep(4.0)

    def land_safely(self) -> None:
        rospy.loginfo("Landing")
        self.led("blink", 0, 255, 80, brightness=160, duration=2.0, priority=8, base=False)
        try:
            self.land()
        except rospy.ServiceException as exc:
            rospy.logerr("Landing service failed: %s", exc)

    def run(self) -> None:
        frame_id = "body"
        try:
            frame_id = self.choose_localization_frame()
            self.takeoff()

            if frame_id == "body":
                self.fly_body_relative_backup_pattern()
            else:
                self.fly_absolute_clover(frame_id)

            self.led("blink", 0, 255, 0, brightness=170, duration=1.5, priority=8, base=False)
            rospy.sleep(0.5)

        except MissionAbort as exc:
            rospy.logerr("Mission aborted: %s", exc)
            self.led("blink_fast", 255, 0, 0, brightness=220, duration=0, priority=20, base=True)
        except rospy.ROSInterruptException:
            rospy.logwarn("ROS interrupt received")
        except Exception as exc:  # keep a real drone from continuing after an unexpected error
            rospy.logerr("Unexpected mission error: %s", exc)
            self.led("blink_fast", 255, 0, 0, brightness=220, duration=0, priority=20, base=True)
        finally:
            self.land_safely()


if __name__ == "__main__":
    mission = CloverComplexPatternMission()
    mission.run()
