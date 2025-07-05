# Import the necessary modules
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas

# Import the config file
from cpg_go1_simulation.config import (
    BLENDER_DATA_DIR,
    BLENDER_HEADERS,
    CONTACT_DATA_DIR,
    CONTROL_DATA_DIR,
    TORQUES_HEADERS,
    TRAJECTORY_PLOT_DIR,
)
from cpg_go1_simulation.sensor.quadruped_robot import QuadrupedRobot
from cpg_go1_simulation.sensor.realtime_controller import RealtimeController


def main(gait_type="trot", walking_mode="plane_straight", use_gui=False, total_time=60):
    """
    Main function to run the simulation.
    Args:
        gait_type: str ("walk", "trot", "pace", "bound", "pronk")
        walking_mode: str ("plane_straight", "eight_ring", "slope_straight")
        use_gui: bool (True, False)
    """
    # Create the robot instance
    robot = QuadrupedRobot(
        robot_name="go1",
        use_gui=use_gui,
        fixed_base=False,
        walking_mode=walking_mode,
    )
    _is_realtime = True if walking_mode == "slope_straight" else False

    # Create the controller instance
    controller = RealtimeController(
        robot,
        gait_type=gait_type,
        total_time=total_time,
        DEBUG=False,
        _is_realtime=_is_realtime,
    )
    # Run the controller
    print("Simulation started")
    control_data, output, torques_data = controller.run(
        gait_type=gait_type, duration=total_time, walking_mode=walking_mode
    )

    # Save trajectory image
    _save_trajectory_image(robot.trajectory_points, walking_mode)

    _save_data(
        BLENDER_DATA_DIR / f"{walking_mode}/To_blender_{gait_type}_data.csv",
        np.array(output),
        headers=BLENDER_HEADERS,
    )
    # Save output data
    if walking_mode == "slope_straight":
        _save_data(
            CONTROL_DATA_DIR
            / f"{walking_mode}/{gait_type}_{walking_mode}_control_data.csv",
            np.array(control_data),
            float_format="%.3f",
        )
        _save_data(
            CONTACT_DATA_DIR / f"{walking_mode}/{gait_type}_torques_data.csv",
            torques_data,
            headers=TORQUES_HEADERS,
        )

    # Reset the robot
    robot.reset()
    print("Simulation ended")


def _save_data(
    file_path: str,
    data: np.array,
    float_format: Optional[str] = None,
    headers: Optional[list[str]] = None,
):
    """Save data to a CSV file
    Args:
        file_path: save file path
        data: the data to be saved
        float_format: str for float formatting
        headers: list[str] for column headers
    """
    # Get the directory path
    directory = file_path.parent
    # Check if the directory exists, if not, create it
    directory.mkdir(parents=True, exist_ok=True)
    df = pandas.DataFrame(data, index=None)
    df.to_csv(
        file_path,
        index=False,
        header=headers if headers else None,
        float_format=float_format,
    )


def _save_trajectory_image(trajectory_points, walking_mode: str = "eight_ring"):
    """Save trajectory image"""
    if trajectory_points:
        # Convert to numpy array
        points = np.array(trajectory_points)

        plt.figure(figsize=(10, 10))
        plt.plot(
            points[:, 0], points[:, 1], "g-", linewidth=2, label="Robot Trajectory"
        )
        plt.scatter(points[0, 0], points[0, 1], color="blue", s=100, label="Start")
        plt.scatter(points[-1, 0], points[-1, 1], color="red", s=100, label="End")

        if walking_mode == "eight_ring":
            # Draw theoretical eight-ring trajectory
            x_right, y_right = eight_ring(3, 5 * np.pi / 6, -np.pi)
            x_right += 3

            x_left, y_left = eight_ring(3, 0, 2 * np.pi - np.pi / 6)
            x_left -= 3

            plt.plot(
                x_right, y_right, "r--", linewidth=2, label="Theoretical Trajectory"
            )
            plt.plot(x_left, y_left, "r--", linewidth=2)

        # Set the image
        plt.title("Robot Trajectory")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid(True)
        plt.axis("equal")  # Maintain aspect ratio
        plt.legend()

        # Save image
        # Get directory path
        directory = TRAJECTORY_PLOT_DIR / f"{walking_mode}"
        # Check if the directory exists, if not, create it
        directory.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{directory}/{walking_mode}_trajectory.png")
        print("Trajectory image has been saved as trajectory.png")
        plt.close()


def eight_ring(radius, start_angle, end_angle):
    """
    Draw eight-ring
    Args:
        radius: Radius
        start_angle: Starting angle
        end_angle: Ending angle
    """
    # Draw circle
    theta = np.linspace(start_angle, end_angle, 1000)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


if __name__ == "__main__":
    # if walking_mode is "eight_ring", gait_type is "trot"
    main(walking_mode="eight_ring", use_gui=True)
