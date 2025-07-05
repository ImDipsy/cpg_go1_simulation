import numpy as np
import pybullet as p

from cpg_go1_simulation.config import JOINT_DATA_PATH


class RobotController:
    """Robot controller for the QuadrupedRobot"""

    def __init__(self, robot):
        """
        Initialize the controller
        Args:
            robot: QuadrupedRobot instance
        """
        self.robot = robot
        self.all_joint_nums = 4 * self.robot.config.joint_nums
        self.last_sensor_data = None

    def control(self, gait_type):
        """Control the robot"""
        foot_position_record = []
        output = []
        linear_velocity = []
        sensor_data_record = []
        link_indices = [i + self.all_joint_nums for i in range(4)]

        # Read control data
        file_path = JOINT_DATA_PATH / f"data_{gait_type}_test.txt"
        with open(file_path, "r") as filestream:
            for line in filestream:
                currentline = line.split(",")
                sim_step = int(currentline[0])  # Get current simulation step
                joints_position = currentline[2:]  # Skip frame and time

                # Set joint positions
                self._set_joint_positions(joints_position)
                p.stepSimulation()

                # Record link positions and foot positions
                link_pos_current = self._record_link_positions(link_indices)
                foot_position_record.append(link_pos_current[1])

                # Record robot state
                output.append(self._record_robot_state(currentline, link_pos_current))

                # Record base velocity
                base_velocity = p.getBaseVelocity(self.robot.robot_id)
                linear_velocity.append(base_velocity[0])

                # Get sensor data
                sensor_data = self.robot.get_sensor_data(sim_step)
                if any(v is not None for v in sensor_data.values()):
                    sensor_data_record.append(sensor_data)

                # time.sleep(1.0 / 2000.0)

        # Output final state
        self._print_final_state(linear_velocity)
        return foot_position_record, sensor_data_record

    def _set_joint_positions(self, joints_position: list[float]):
        """Set joint positions"""
        for j in range(self.all_joint_nums):
            target_pos = float(joints_position[j])
            p.setJointMotorControl2(
                self.robot.robot_id,
                self.robot.joint_ids[j],
                p.POSITION_CONTROL,
                self.robot.config.joint_directions[j] * target_pos
                + self.robot.config.joint_offsets[j],
                force=self.robot.config.max_forces[j],
            )

    def _record_link_positions(self, link_indices: list[int]):
        """Record link positions"""
        return [p.getLinkState(self.robot.robot_id, index)[0] for index in link_indices]

    def _record_robot_state(
        self, currentline: list[float], link_pos_current: list[float]
    ):
        """Record robot state"""
        current_time = [int(currentline[0]) / 500]
        body_pos = list(p.getBasePositionAndOrientation(self.robot.robot_id)[0])
        euler = list(
            p.getEulerFromQuaternion(
                p.getBasePositionAndOrientation(self.robot.robot_id)[1]
            )
        )
        joint_pos = [
            p.getJointStates(self.robot.robot_id, range(12))[i][0] for i in range(12)
        ]
        return current_time + body_pos + euler + joint_pos + list(link_pos_current[1])

    def _print_final_state(self, linear_velocity):
        """Print final state"""
        body_pos = list(p.getBasePositionAndOrientation(self.robot.robot_id)[0])
        mean_velocity = np.mean(np.array(linear_velocity), axis=0)
        print("Body Position:", self._decimal(body_pos))
        print("Mean Velocity:", self._decimal(mean_velocity))
        print("Velocity/Body Length:", self._decimal(mean_velocity / 0.217))

    @staticmethod
    def _decimal(data, num=3):
        """Control the precision of floating-point numbers."""
        return np.array([round(i, num) for i in data]).astype(float)
