import os

# Get the project root directory
from dataclasses import dataclass

import cv2
import numpy as np
import pybullet as p
import pybullet_data as pd

from cpg_go1_simulation.config import OUTPUT_IMAGE_DIR
from cpg_go1_simulation.sensor.sensor import IMU, ContactSensor, DepthCamera


@dataclass
class RobotConfig:
    """Configuration of the robot"""

    name: str
    urdf_path: str
    joint_offsets: list
    joint_directions: list
    max_forces: list
    joint_nums: int
    initial_pos: list
    base_position: list = None
    base_orientation: list = None
    walking_mode: str = "eight_ring"


class QuadrupedRobot:
    """Quadruped robot class"""

    def __init__(
        self,
        robot_name="go1",
        use_gui=True,
        fixed_base=False,
        walking_mode="eight_ring",
    ):
        """
        Initialize the quadruped robot
        Args:
            robot_name: Robot name
            use_gui: Whether to use GUI
            fixed_base: Whether to fix the robot base
            walking_mode: "eight_ring","plane_straight","slope_straight"
        """
        self.robot_name = robot_name
        self.use_gui = use_gui
        self.fixed_base = fixed_base
        self.walking_mode = walking_mode

        # Eight ring parameters
        self.eight_params = {
            "radius": 3.0,
            "height": 0.0,
            "start_angle": 5 * np.pi / 6,  # Starting angle
        }

        # Initialize configuration and environment
        self._init_robot_config()
        self._setup_environment()

        # Initialize sensors
        self.imu = IMU(self.robot_id)
        self.camera = DepthCamera(self.robot_id)
        self.contact_sensors = ContactSensor(self.robot_id, self.plane_id)
        # Image counter
        self.image_counter = 0

    def _init_robot_config(self):
        """Initialize robot configuration"""
        # Calculate eight ring start position and orientation
        if self.walking_mode == "eight_ring":
            start_pos, start_orn = self._calculate_eight_start_pose()
        else:
            start_pos = [0, 0, 0.28]
            start_orn = [0, 0, 0, 1]

        configs = {
            "go1": RobotConfig(
                name="go1",
                urdf_path="Quadruped_model/go1/go1.urdf",
                joint_offsets=[0] * 12,
                joint_directions=[-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1],
                max_forces=[23.7, 23.7, 35.55] * 4,
                joint_nums=3,
                initial_pos=[0, 0.6, -1.3] * 4,
                base_position=start_pos,
                base_orientation=start_orn,
                walking_mode=self.walking_mode,
            ),
        }
        self.config = configs.get(self.robot_name)
        if not self.config:
            raise ValueError(f"Unsupported robot type: {self.robot_name}")

    def _calculate_eight_start_pose(self):
        """Calculate eight ring start position and orientation"""
        r = self.eight_params["radius"]
        h = self.eight_params["height"]
        theta = self.eight_params["start_angle"]

        # Calculate start position (on the right circle)
        x = r + r * np.cos(theta)
        y = r * np.sin(theta)
        z = h + 0.35  # Add the height of the robot off the ground

        tangent_x = -r * np.sin(theta)
        tangent_y = r * np.cos(theta)

        # Calculate the orientation quaternion (to make the robot face the tangent direction)
        yaw = np.arctan2(tangent_y, tangent_x)  # Updated to the correct calculation
        orientation = p.getQuaternionFromEuler([0, 0, yaw + np.pi])

        return [x, y, z], orientation

    def _setup_environment(self):
        """Set up the simulation environment"""
        # Set working directory and connect to the simulator
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        p.connect(p.GUI if self.use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pd.getDataPath())

        # Set physical parameters
        p.setPhysicsEngineParameter(
            enableConeFriction=1, deterministicOverlappingPairs=1
        )
        p.setDefaultContactERP(0)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 500)

        # Load the scene
        self.plane_id = self._load_ground()

        # Load the robot
        self.robot_id = self._load_robot()
        self.joint_ids = self._get_joint_ids()
        self._set_dynamics()

        # Set debug camera view (looking straight down)
        if self.use_gui:
            if self.walking_mode == "eight_ring":
                p.resetDebugVisualizerCamera(
                    cameraDistance=5.0,  # Camera distance
                    cameraYaw=0,  # Camera yaw angle (left and right)
                    cameraPitch=-89,  # Camera pitch angle (up and down, -90 for straight down)
                    cameraTargetPosition=[0, 0, 0],  # Camera target position
                )

            elif self.walking_mode == "slope_straight":
                p.resetDebugVisualizerCamera(
                    cameraDistance=8.0,  # Camera distance
                    cameraYaw=-45,  # Camera yaw angle (left and right)
                    cameraPitch=0,  # Camera pitch angle (up and down, -90 for straight down)
                    cameraTargetPosition=[5, 0, 0],  # Camera target position
                )

    def _load_ground(self):
        """Load the ground"""
        if self.walking_mode == "slope_straight":
            plane_id = p.loadURDF(
                "Quadruped_model/terrain/part_terrain_short.urdf",
                [3, -1, 0],
            )
        elif self.walking_mode == "eight_ring":
            plane_id = p.loadURDF("Quadruped_model/plane/plane10.urdf")
            self._create_figure_eight()
        elif self.walking_mode == "plane_straight":
            plane_id = p.loadURDF("Quadruped_model/plane/plane1000.urdf")
        return plane_id

    def _create_figure_eight(self):
        """Create a figure-eight trajectory"""
        r = self.eight_params["radius"]
        h = self.eight_params["height"]
        tube_radius = 0.01

        def _create_torus_segment(center, start_angle, end_angle, num_segments=100):
            # Number of segments for the torus segment
            theta = np.linspace(start_angle, end_angle, num_segments)

            # Create the torus segment
            for i in range(len(theta) - 1):
                # Calculate the start and end points for the current segment
                p1 = np.array(
                    [
                        center[0] + r * np.cos(theta[i]),
                        center[1] + r * np.sin(theta[i]),
                        h,
                    ]
                )
                p2 = np.array(
                    [
                        center[0] + r * np.cos(theta[i + 1]),
                        center[1] + r * np.sin(theta[i + 1]),
                        h,
                    ]
                )

                # Calculate the direction and length of the segment
                direction = p2 - p1
                length = np.linalg.norm(direction)

                # Calculate the midpoint and normalized direction
                mid_point = (p1 + p2) / 2
                direction_normalized = direction / length

                # Calculate the quaternion for rotation
                up_vector = np.array([0, 0, 1])
                rotation_axis = np.cross(up_vector, direction_normalized)
                rotation_angle = np.arccos(np.dot(up_vector, direction_normalized))

                if np.linalg.norm(rotation_axis) > 0:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    quat = p.getQuaternionFromAxisAngle(rotation_axis, rotation_angle)
                else:
                    quat = [0, 0, 0, 1]

                # Create the visual shape without a collision shape
                visual_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_CYLINDER,
                    radius=tube_radius,
                    length=length,
                    rgbaColor=[0, 0, 0, 0.8],  # Black
                )

                # Create a non-colliding multibody
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,  # -1 means no collision shape
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=mid_point,
                    baseOrientation=quat,
                )

        # Create right torus
        _create_torus_segment([r, 0, 0], self.eight_params["start_angle"], -np.pi)

        # Create left torus
        _create_torus_segment([-r, 0, 0], 0, 2 * np.pi - np.pi / 6)

    def _load_robot(self):
        """Load the robot"""
        return p.loadURDF(
            self.config.urdf_path,
            self.config.base_position,
            self.config.base_orientation,
            flags=p.URDF_USE_INERTIA_FROM_FILE + p.URDF_MAINTAIN_LINK_ORDER,
            useFixedBase=self.fixed_base,
        )

    def _get_joint_ids(self):
        """Get the joint IDs"""
        return [
            j
            for j in range(p.getNumJoints(self.robot_id))
            if p.getJointInfo(self.robot_id, j)[2]
            in (p.JOINT_PRISMATIC, p.JOINT_REVOLUTE)
        ]

    def _set_dynamics(self):
        """Set dynamics parameters"""
        # Set joint parameters
        for i in self.joint_ids:
            p.changeDynamics(
                self.robot_id,
                i,
                linearDamping=0.0,
                angularDamping=0.0,
                lateralFriction=0.2,
                rollingFriction=0.1,
                restitution=0.999,
                frictionAnchor=0,
            )

        # Set end effector parameters
        lateral_Fric = 1.0 if self.walking_mode == "slope_straight" else 0.6

        for i in range(max(self.joint_ids) + 1, max(self.joint_ids) + 5):
            p.changeDynamics(
                self.robot_id,
                i,
                linearDamping=0.0,
                angularDamping=0.0,
                lateralFriction=lateral_Fric,
                restitution=0.999,
                frictionAnchor=0,
            )

    def reset(self):
        """Set the robot back to the initial state"""
        # Set joint states
        for i, joint_id in enumerate(self.joint_ids):
            p.resetJointState(
                self.robot_id,
                joint_id,
                targetValue=self.config.initial_pos[i] * self.config.joint_directions[i]
                + self.config.joint_offsets[i],
                targetVelocity=0,
            )

        # Reset the base position and orientation
        p.resetBasePositionAndOrientation(
            self.robot_id, self.config.base_position, self.config.base_orientation
        )
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
        p.stepSimulation()

        # Clear trajectory
        if hasattr(self, "trajectory_visual_ids"):
            for point_id in self.trajectory_visual_ids:
                p.removeBody(point_id)
            self.trajectory_points = []
            self.trajectory_visual_ids = []

    def get_sensor_data(self):
        """Get sensor data"""
        # Get image data when walking mode is eight ring
        rgb_image, depth_image = self.camera.get_image()

        return {"rgb_image": rgb_image, "depth_image": depth_image}

    def _save_rgb_image(self, rgb_array: np.ndarray) -> bool:
        """Save the RGB image to png file"""
        if rgb_array is None:
            print("Cannot save RGB image: No image data")
            return False

        try:
            # Convert RGBA format to RGB format
            rgb_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

            # Ensure directory exists
            if not os.path.exists(OUTPUT_IMAGE_DIR):
                os.makedirs(OUTPUT_IMAGE_DIR)

            # Generate four-digit number
            image_number = f"{self.image_counter:04d}"

            # Construct file path of the image
            file_path = OUTPUT_IMAGE_DIR / f"rgb_image{image_number}.png"

            # Save image
            success = cv2.imwrite(file_path, rgb_image)
            print(f"RGB image saved as {file_path}")
            if not success:
                print(f"Failed to save RGB image to {file_path}")
                return False

            self.image_counter += 1
            return True

        except Exception as e:
            print(f"Error occurred while saving RGB image: {str(e)}")
            return False

    def _track_trajectory(self):
        """Record and visualize the robot's trajectory"""
        if not hasattr(self, "trajectory_points"):
            self.trajectory_points = []
            self.trajectory_visual_ids = []

        # Get the robot's center of mass position
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        self.trajectory_points.append(pos)

        # if self.use_gui:
        #     # Visualize the trajectory
        #     if len(self.trajectory_points) >= 2:
        #         p.addUserDebugLine(
        #             self.trajectory_points[-2],
        #             self.trajectory_points[-1],
        #             lineColorRGB=[0, 1, 0],
        #             lineWidth=5,
        #         )
