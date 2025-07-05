import numpy as np
import pybullet as p


class IMU:
    """IMU sensor class"""

    def __init__(self, robot_id):
        self.robot_id = robot_id
        # Noise parameters
        self.gyro_noise = 0.001
        self.accel_noise = 0.001

    def get_orientation(self) -> list[float]:
        """Get robot orientation"""
        _, quat = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(quat)
        euler_with_noise = [e + np.random.normal(0, self.gyro_noise) for e in euler]
        return euler_with_noise

    def get_position(self) -> list[float]:
        """Get robot position"""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        # Add noise
        pos_with_noise = [p + np.random.normal(0, self.accel_noise) for p in pos]
        return pos_with_noise

    def get_velocity(self):
        """Get robot velocity"""
        linear_vel, angular_vel = p.getBaseVelocity(self.robot_id)
        # Add noise
        linear_vel_with_noise = [
            v + np.random.normal(0, self.accel_noise) for v in linear_vel
        ]
        angular_vel_with_noise = [
            v + np.random.normal(0, self.gyro_noise) for v in angular_vel
        ]
        return linear_vel_with_noise, angular_vel_with_noise


class DepthCamera:
    """Depth camera class"""

    def __init__(self, robot_id, camera_position=[0.2, 0, 0.05]):
        self.robot_id = robot_id
        self.camera_position = camera_position
        # Reduce image resolution
        self.width = 128
        self.height = 128
        self.fov = 60
        self.aspect = self.width / self.height
        self.nearVal = 0.01
        self.far = 3.0

        # Cache projection matrix to avoid redundant calculations
        self.proj_mat = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect,
            nearVal=self.nearVal,
            farVal=self.far,
        )
        self.image_counter = 0

    def set_camera(self, robot_id: int, width: int = 128, height: int = 128):
        position, orientation = p.getBasePositionAndOrientation(robot_id)
        r_mat = p.getMatrixFromQuaternion(orientation)

        # Calculate the vectors
        tx_vec = np.array([r_mat[0], r_mat[3], r_mat[6]])  # Forward vector
        # ty_vec = np.array([r_mat[1], r_mat[4], r_mat[7]])  # Left vector
        tz_vec = np.array([r_mat[2], r_mat[5], r_mat[8]])  # Up vector

        # Calculate the camera position: slightly above and in front of the robot
        camera_position = np.array(position) + 0.3 * tx_vec + 0.05 * tz_vec

        # Calculate the target position: slightly downward in front of the camera
        target_position = camera_position + tx_vec - 0.2 * tz_vec

        # Calculate the view matrix
        view_mat = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=target_position,  # Look slightly downward in front
            cameraUpVector=tz_vec,  # Keep camera facing up
        )

        # Use a faster renderer
        w, h, rgb, depth, seg = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_mat,
            projectionMatrix=self.proj_mat,
            renderer=p.ER_TINY_RENDERER,  # Use a faster renderer
        )
        return w, h, rgb, depth, seg

    def get_image(self):
        """Get depth image and RGB image"""
        try:
            w, h, rgb, depth, seg = self.set_camera(self.robot_id)

            # Ensure data is not empty
            if depth is None or rgb is None:
                print("Warning: Camera data retrieval failed")
                return None, None

            # Convert to numpy array and reshape
            rgb_array = np.array(rgb).astype("uint8")
            if len(rgb_array.shape) == 1:  # If it's a one-dimensional array
                rgb_array = rgb_array.reshape(h, w, 4)  # RGBA format

            depth_array = np.array(depth).astype("uint8")
            if len(depth_array.shape) == 1:  # If it's a one-dimensional array
                depth_array = depth_array.reshape(h, w)

            # Check if arrays are empty
            if depth_array.size == 0 or rgb_array.size == 0:
                print("Warning: Empty image data")
                return None, None

            # Normalize depth image
            depth_range = depth_array.max() - depth_array.min()
            if depth_range == 0:
                depth_normalized = np.zeros_like(depth_array)
            else:
                depth_normalized = (depth_array - depth_array.min()) / depth_range

            return rgb_array, depth_normalized

        except Exception as e:
            print(f"Camera data processing error: {str(e)}")
            print(f"RGB shape: {rgb_array.shape if 'rgb_array' in locals() else 'N/A'}")
            print(
                f"Depth shape: {depth_array.shape if 'depth_array' in locals() else 'N/A'}"
            )
            return None, None


class ContactSensor:
    def __init__(self, robot_id, plane_id):
        self.robot_id = robot_id
        self.plane_id = plane_id

    def get_contact_info(self, link_indices: list[int]) -> dict[str, list[float]]:
        """Detect contact information between the robot and the ground plane.
        Args:
            link_indices (list[int]): The link indices to check for contact.
        Returns:
            dict[str, list[float]]: The contact information.
        """

        contact_points = p.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id)

        # Initialize the dictionary to store results
        contact_info = {}

        for contact_link in link_indices:
            contact_info[f"normal_vector_{contact_link}"] = None

        for point in contact_points:
            # Check if the contacted link is one of the link indices
            if point[3] in link_indices or point[4] in link_indices:
                # Get the direction of the normal force
                normal_vector = point[7]
                contact_info[f"normal_vector_{point[3]}"] = normal_vector
            else:
                # If the contacted link is not in link indices, skip
                continue

            # Get the direction of the normal force
            contact_info[f"normal_vector_{point[3]}"] = normal_vector
        return contact_info
