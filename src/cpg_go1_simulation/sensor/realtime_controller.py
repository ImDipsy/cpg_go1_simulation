import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas
import pybullet as p
import torch

from cpg_go1_simulation.config import BEST_MODEL_PATH, GAIT_DATA_DIR
from cpg_go1_simulation.execution_neural_network.mlp import MLP
from cpg_go1_simulation.gait_and_transition.export_cpg import export_cpg_data
from cpg_go1_simulation.sensor.reflexion_processor import ReflexionProcessor
from cpg_go1_simulation.sensor.robot_controller import RobotController
from cpg_go1_simulation.sensor.visual_processor_rgb import PathTracker
from cpg_go1_simulation.stein.implementations import CPG8Neuron


@dataclass
class RealtimeConfig:
    """Configurations for the realtime controller"""

    # Control frequency (Hz)
    CONTROL_FREQ: float = 500.0
    # Visual processing frequency (Hz)
    VISUAL_FREQ: float = 50.0
    # Visual processing interval steps
    VISUAL_INTERVAL: int = int(CONTROL_FREQ / VISUAL_FREQ)

    # IMU frequency (Hz)
    IMU_FREQ: float = 250.0
    # IMU processing interval steps
    IMU_INTERVAL: int = int(CONTROL_FREQ / IMU_FREQ)

    # CPG frequency (Hz)
    CPG_FREQ: float = 500.0
    # Control parameters
    # Smoothing window size
    SMOOTHING_WINDOW: int = 5
    # Minimum and maximum velocity ratios
    MIN_RATIO: float = 0.6
    MAX_RATIO: float = 1.2
    # Default velocity ratio
    DEFAULT_RATIO: float = 0.8
    # Start record time (s)
    START_RECORD_TIME: float = 1.0

    # Gait mapping
    GAIT_MAP = {"walk": 1, "trot": 2, "pace": 3, "bound": 4, "pronk": 5}


class RealtimeController:
    """Realtime controller for the quadruped robot"""

    def __init__(
        self,
        robot,
        gait_type: str = "trot",
        total_time: float = 60.0,
        DEBUG: bool = False,
        _is_realtime: bool = False,
    ):
        self.total_time = total_time
        self.robot = robot
        # if DEBUG is True, print debug information
        self.DEBUG = DEBUG

        # if _is_realtime is True, the controller will use the reflexion processor
        self._is_realtime = _is_realtime
        # time mask: wait until the CPG reaches time_mask to start sending data (in seconds)
        self.time_mask = 2.0
        # Load the configuration
        self.config = RealtimeConfig()
        # Create the base controller
        self.base_controller = RobotController(robot)

        # Create the visual processor
        self.path_tracker = PathTracker(DEBUG=self.DEBUG)

        # Create the CPG processor: it must be created before the reflexion processor
        self.cpg_processor = CPG8Neuron(
            before_ftype=self.config.GAIT_MAP[gait_type],
            after_ftype=self.config.GAIT_MAP[gait_type],
            total_time=0.002,
            toc=10.0,
        )

        # Create the reflexion processor
        self.reflexion_processor = ReflexionProcessor(
            initial_gait=gait_type, time_mask=self.time_mask
        )

        # Load the neural network model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.version = self._load_model()

        # State records
        self.velocity_ratios = []
        self.errors = []
        self.current_ratios = [self.config.DEFAULT_RATIO] * 8
        self.current_error = 0
        self.sim_step = 0
        self.current_time = 0

        # CPG data
        self.cpg_data = None

        # Joint mapping
        self.joint_mapping = {
            0: [1, 2],  # First prediction maps to joints 1, 2
            1: [4, 5],  # Second prediction maps to joints 4, 5
            2: [7, 8],  # Third prediction maps to joints 7, 8
            3: [10, 11],  # Fourth prediction maps to joints 10, 11
        }

        # Data recording lists
        self.error_log = []
        self.ratio_log = []

    def _load_model(self):
        """Load the trained model"""
        model_path = BEST_MODEL_PATH

        try:
            # Check if the file exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model file does not exist: {model_path}")

            # Load the saved data
            logging.info(f"Loading model: {model_path}")
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            # Get version number
            version = checkpoint.get("model_config", {}).get("version", "1.2")
            logging.info(f"Model version: v{version}")

            # Create model
            model = MLP()

            # Validate checkpoint format
            required_keys = [
                "model_state_dict",
                "model_config",
                "training_config",
            ]
            for key in required_keys:
                if key not in checkpoint:
                    raise KeyError(f"Checkpoint is missing required key: {key}")

            # Load model state
            try:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                logging.info("Model state loaded successfully")
            except Exception as e:
                logging.warning(f"Warning occurred while loading model state: {e}")

            # Log model configuration information
            if "model_config" in checkpoint:
                logging.info(f"Model configuration: {checkpoint['model_config']}")
            if "training_config" in checkpoint:
                logging.info(f"Training configuration: {checkpoint['training_config']}")
            if "best_val_loss" in checkpoint:
                logging.info(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
            if "epoch" in checkpoint:
                logging.info(f"Training epoch: {checkpoint['epoch'] + 1}")

            model = model.to(self.device)
            model.eval()
            logging.info("Model loaded successfully")
            return model, version

        except Exception as e:
            logging.error(f"Error occurred while loading model: {e}")
            raise

    def _load_cpg_data(self, gait_type: str, total_time: float = 125.0):
        """Load CPG data
        Args:
            gait_type: Gait type
        """
        file_path = GAIT_DATA_DIR / f"cpg_{gait_type}_{total_time}s.csv"
        if not file_path.exists():
            print(f"Cpg data not found, generating new data for {gait_type}")
            export_cpg_data(gait_type=gait_type, total_time=total_time)

        self.cpg_data = pandas.read_csv(file_path, header=None).values

        print(f"CPG data loaded for {gait_type}")

        start_idx = int(0 * self.config.CPG_FREQ)
        self.cpg_data = self.cpg_data[start_idx:]

    def _process_data(self, cpg_data: np.ndarray, gait_type: str) -> list:
        """Process CPG data to generate network input"""
        processed_data = []
        onehot_gait = self._get_gait_onehot(gait_type)

        for i in range(8):
            if gait_type == "walk":
                current_ratio = self.current_ratios[i] + 0.1
            else:
                current_ratio = self.current_ratios[i]
            features = np.concatenate(
                [
                    self._get_joint_onehot(i),
                    onehot_gait,
                    [current_ratio],  # Use the corresponding neuron's velocity ratio
                    [cpg_data[0, i]],  # Value of the i-th neuron
                    [cpg_data[0, i + 8] * 1 / 50],  # Derivative of the i-th neuron
                ]
            )
            processed_data.append(features)

        return processed_data

    def _get_joint_onehot(self, joint_idx: int) -> list:
        """Get joint encoding"""
        encodings = [
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0],
        ]
        return encodings[joint_idx]

    def _get_gait_onehot(self, gait_type: str) -> dict:
        """Get gait encoding"""
        gait_map = {
            "walk": [1, 0, 0, 0, 0],
            "trot": [0, 1, 0, 0, 0],
            "pace": [0, 0, 1, 0, 0],
            "bound": [0, 0, 0, 1, 0],
            "pronk": [0, 0, 0, 0, 1],
        }
        return gait_map[gait_type]

    def _predict_joint_angles(self, processed_data: np.ndarray) -> np.ndarray:
        """Predict joint angles"""
        with torch.no_grad():
            input_tensor = torch.FloatTensor(processed_data).to(self.device)
            predictions = self.model(input_tensor).cpu().numpy()

        # Map 8 predictions to 12 joints
        joint_angles = np.zeros(
            12
        )  # Initialize joint angles for 12 joints, keeping 0,3,6,9 as 0

        # Mapping: network output -> URDF joint index
        mapping = {
            0: 1,  # Predictions 0 -> Joint 1
            1: 4,  # Predictions 1 -> Joint 4
            2: 7,  # Predictions 2 -> Joint 7
            3: 10,  # Predictions 3 -> Joint 10
            4: 2,  # Predictions 4 -> Joint 2
            5: 5,  # Predictions 5 -> Joint 5
            6: 8,  # Predictions 6 -> Joint 8
            7: 11,  # Predictions 7 -> Joint 11
        }

        # Apply mapping
        predictions = predictions.reshape(-1)  # Ensure it's a 1D array
        for pred_idx, joint_idx in mapping.items():
            joint_angles[joint_idx] = predictions[pred_idx]

        return joint_angles

    def _update_velocity_ratio(self, new_ratios: list):
        """Update the velocity ratios"""
        self.velocity_ratios.append(new_ratios)
        if len(self.velocity_ratios) > self.config.SMOOTHING_WINDOW:
            self.velocity_ratios.pop(0)

        # Compute smoothed ratios for each neuron
        if self.velocity_ratios:
            smoothed_ratios = np.mean(self.velocity_ratios, axis=0)
            # Ensure each ratio is within the allowed range
            self.current_ratios = np.clip(
                smoothed_ratios, self.config.MIN_RATIO, self.config.MAX_RATIO
            ).tolist()
        else:
            self.current_ratios = [self.config.DEFAULT_RATIO] * 8

    # Get the joint torques
    def _get_joint_torques(self, joint_indices: list) -> list[float]:
        """
        Get the joint torques with the given joint indices

        Args:
            robot_id (int): Robot ID.
            joint_indices (list):  List of joint indices to retrieve torques for.

        Returns:
            list: List of torques corresponding to the provided joint indices.
        """
        joint_torques = []
        for joint_id in joint_indices:
            # Get joint state, where [3] is the joint torque
            joint_state = p.getJointState(self.robot.robot_id, joint_id)
            torque = joint_state[3]
            joint_torques.append(torque)
        return joint_torques

    def _get_body_and_joint_data(self):
        """Get robot body and joint data"""
        body_pos = list(p.getBasePositionAndOrientation(self.robot.robot_id)[0])
        body_euler = list(
            p.getEulerFromQuaternion(
                p.getBasePositionAndOrientation(self.robot.robot_id)[1]
            )
        )
        joint_pos = [
            p.getJointStates(self.robot.robot_id, range(12))[i][0] for i in range(12)
        ]

        return body_pos, body_euler, joint_pos

    def control_step(
        self, gait_type: str, walking_mode: str = "plane_straight"
    ) -> Dict:
        """Execute a control step"""
        # 1. Retrieve joint torques
        joint_torques = self._get_joint_torques([1, 4, 7, 10])
        sum_torques = sum(joint_torques)

        # if it is not real-time, load the CPG data
        if not self._is_realtime:
            # 2.1. Make sure CPG data is loaded
            if self.cpg_data is None:
                self._load_cpg_data(gait_type=gait_type, total_time=self.total_time)

            # 2.2. Get the current CPG state
            current_step = self.sim_step % len(self.cpg_data)
            cpg_state = self.cpg_data[current_step : current_step + 1]
            current_gait = gait_type
        else:
            # 2.3. Update the CPG data
            # Update CPG state and gait
            current_gait, cpg_state, avg_torque = self.reflexion_processor.update(
                sim_time=self.current_time, sum_torque=sum_torques
            )

            print(
                "Current Time: ",
                self.current_time,
                f"Current Gait: {current_gait}",
            )

        # 3. Process sensor information and update velocity ratios
        if walking_mode == "slope_straight":
            if self.sim_step % self.config.IMU_INTERVAL == 0:
                # Update velocity ratios
                orientation = self.robot.imu.get_orientation()
                imu_data = {"orientation": orientation}
                error, ratio = self.path_tracker.process_straight_walking(imu_data)
                self._update_velocity_ratio(ratio)

        # Plane straight walking mode
        elif walking_mode == "plane_straight":
            if self.sim_step % self.config.IMU_INTERVAL == 0:
                # Straight walking mode
                orientation = self.robot.imu.get_orientation()
                imu_data = {"orientation": orientation}
                error, ratio = self.path_tracker.process_straight_walking(imu_data)
                self._update_velocity_ratio(ratio)

        # Eight ring tracking mode
        elif walking_mode == "eight_ring":
            if self.sim_step % self.config.VISUAL_INTERVAL == 0:
                sensor_info = self.robot.get_sensor_data()
                if sensor_info["rgb_image"] is not None:
                    error, ratio, stopFlag = self.path_tracker.process_frame(
                        sensor_info["rgb_image"]
                    )
                    self.current_error = error
                    if stopFlag:
                        return None
                    self._update_velocity_ratio(ratio)
                    # If in debug mode, print information
                    if self.DEBUG:
                        print(
                            f"Eight ring tracking - Error: {error:.3f}, Ratios: {ratio}"
                        )
                    self.robot._save_rgb_image(sensor_info["rgb_image"])

        else:
            raise ValueError(f"Unsupported walking mode: {walking_mode}")

        # 4. Process CPG data
        processed_data = self._process_data(cpg_state, current_gait)

        # 5. Predict joint angles
        joint_angles = self._predict_joint_angles(processed_data)

        # 6. Set joint positions
        self.base_controller._set_joint_positions(joint_angles.tolist())
        p.stepSimulation()

        # 7. Update counter and timestamp
        self.sim_step += 1
        self.current_time += 1 / self.config.CONTROL_FREQ
        self.current_time = round(self.current_time, 5)

        # Print current simulation time and show progress
        print(f"Current simulation step: {self.sim_step}")

        # 8. Track trajectory
        self.robot._track_trajectory()

        # 9. Return status information
        base_state = {
            "current_gait": current_gait,
            "sim_step": self.sim_step,
            "joint_angles": joint_angles,
            "velocity_ratio": self.current_ratios,
            "error": self.current_error,
            "cpg_state": cpg_state.reshape(-1),
            "sum_joint_torques": sum_torques,
        }

        # If avg_torque exists, add it to the state information
        if "avg_torque" in locals():
            base_state["avg_joint_torques"] = avg_torque
        return base_state

    def run(self, gait_type: str, duration: float, walking_mode: str = "eight_ring"):
        """Run the controller
        Args:
            gait_type: Type of gait
            duration: Running duration (seconds)
            walking_mode: Walking mode, "eight_ring" for eight-ring path tracking, "plane_straight" for straight walking
        """
        total_steps = int(duration * self.config.CONTROL_FREQ)
        control_data = []
        torques_data = []
        output = []

        for _ in range(total_steps):
            state = self.control_step(gait_type, walking_mode)
            if state is None:
                break
            control_data.append(
                [state["sim_step"] / self.config.CONTROL_FREQ]
                + [state["error"]]
                + state["velocity_ratio"]
            )
            body_pos, body_euler, joint_pos = self._get_body_and_joint_data()
            thigh_joint_torques = self._get_joint_torques([1, 4, 7, 10])

            # If avg_joint_torques exists, add it to the torques_data
            if walking_mode == "slope_straight":
                torques_data.append(
                    [state["sim_step"] / self.config.CONTROL_FREQ]
                    + body_pos
                    + body_euler
                    + thigh_joint_torques
                    + [state["sum_joint_torques"]]
                    + [state["current_gait"]]
                    + [state["avg_joint_torques"]]
                )
            output.append(
                [state["sim_step"] / self.config.CONTROL_FREQ]
                + body_pos
                + body_euler
                + joint_pos
            )

        for i in range(len(self.error_log)):
            step, err = self.error_log[i]
            _, r1, r2 = self.ratio_log[i]
            print(f"{step}\t{err:.3f}\t{r1:.3f}\t{r2:.3f}")

        return control_data, output, torques_data
