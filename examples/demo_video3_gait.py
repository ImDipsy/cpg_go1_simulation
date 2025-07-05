import math
import time
from typing import Optional

import pandas
import pybullet as p
import pybullet_data as pd

from cpg_go1_simulation.config import (
    BLENDER_DATA_DIR,
    BLENDER_HEADERS,
    GAIT_DATA_DIR,
    ROBOT_MODEL_DIR,
    TRANSITION_DATA_DIR,
)
from cpg_go1_simulation.gait_and_transition.export_cpg import export_cpg_data
from cpg_go1_simulation.gait_and_transition.process_network import ProcessNetwork


def gait(
    gait_type: str = "walk",
    total_time: float = 25,
    use_gui: bool = True,
    command_time: Optional[float] = 15.0,
):
    """Execute gait or gait transition simulation

    Args:
        gait_type (str): The gait type to simulate. Defaults to 'walk'.
        total_time (float): The total time for the simulation. Defaults to 25.
        use_gui (bool): Whether to use the GUI. Defaults to True.
    """
    # Check and get CPG data file path
    if "_to_" in gait_type:
        formatted_command_time = "{:.3f}".format(command_time)
        # Handle transition gait
        transition_dir = TRANSITION_DATA_DIR
        # Find matching files
        pattern = f"cpg_{gait_type}_{formatted_command_time}_*.csv"
        matching_files = list(transition_dir.glob(pattern))

        if matching_files:
            # If file found, use the first matching file, or you can change the logic to use the latest file
            file_path = matching_files[0]
            # Extract toc1 and toc2 from filename
            # Filename format: cpg_walk_to_trot_15.000_15.000.csv
            filename = file_path.stem  # Get filename without extension
            toc_values = filename.split("_")[-2:]  # Get last two values
            execution_time = float(toc_values[1])
        else:
            # If no file found, use default values and generate new file
            execution_time = export_cpg_data(
                gait_type=gait_type, command_time=command_time, total_time=total_time
            )
            formatted_execution_time = "{:.3f}".format(
                (math.ceil(execution_time * 500) * 0.002)
            )
            # Get the file path
            file_path = (
                TRANSITION_DATA_DIR
                / f"cpg_{gait_type}_{formatted_command_time}_{formatted_execution_time}.csv"
            )

    else:
        # Handle single gait
        file_path = GAIT_DATA_DIR / f"cpg_{gait_type}_{total_time}s.csv"
        if not file_path.exists():
            export_cpg_data(gait_type=gait_type, total_time=total_time)

    # Ensure file exists
    if not file_path.exists():
        raise FileNotFoundError(f"CPG data file not found: {file_path}")

    else:
        df = pandas.read_csv(file_path, header=None)
        cpg_data = df.to_numpy()

    if use_gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())

    p.setPhysicsEngineParameter(enableConeFriction=1, deterministicOverlappingPairs=1)
    p.setDefaultContactERP(0)
    p.setPhysicsEngineParameter(enableConeFriction=0)

    plane_path = ROBOT_MODEL_DIR / "plane/plane1000.urdf"
    plane_path_str = plane_path.as_posix()
    plane = p.loadURDF(plane_path_str)

    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1.0 / 500)
    # p.setDefaultContactERP(0)
    # urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
    urdfFlags = p.URDF_USE_INERTIA_FROM_FILE + p.URDF_MAINTAIN_LINK_ORDER

    # Set the robot model and initial height
    initial_height = 0.25
    robot_path = ROBOT_MODEL_DIR / "go1/go1.urdf"
    robot_path_str = robot_path.as_posix()
    quadruped = p.loadURDF(
        robot_path_str,
        [0, 0, initial_height],
        [0, 0, 0, 1],
        flags=urdfFlags,
        useFixedBase=False,
    )
    #  useFixedBase=True)

    jointIds = []
    for j in range(p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
        info = p.getJointInfo(quadruped, j)
        jointName = info[1]
        jointType = info[2]
        if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
            jointIds.append(j)

    # Set the dynamics parameters of the robot
    for i in jointIds:
        p.changeDynamics(
            quadruped,
            i,
            linearDamping=0.0,
            angularDamping=0.0,
            lateralFriction=0.2,
            rollingFriction=0.1,
            restitution=0.999,
            frictionAnchor=0,
        )
    # Set the dynamics parameters of the robot
    for i in range(12, 16):
        p.changeDynamics(
            quadruped,
            i,
            linearDamping=0.0,
            angularDamping=0.0,
            lateralFriction=0.6,
            restitution=0.999,
            frictionAnchor=0,
        )

    jointOffsets = []
    for i in range(4):
        jointOffsets.append(0)
        jointOffsets.append(0)
        jointOffsets.append(0)

    for i in range(12):
        p.resetJointState(
            bodyUniqueId=quadruped,
            jointIndex=jointIds[i],
            targetValue=jointOffsets[i],
            targetVelocity=0,
        )

    jointDirections = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]

    maxForceId = []
    for i in range(4):
        maxForceId.append(23.7)
        maxForceId.append(23.7)
        maxForceId.append(35.55)

    p.getCameraImage(480, 320)
    p.setRealTimeSimulation(0)

    # Wait for the robot to fall to the ground
    for _ in range(int(0.2 * 500)):
        p.stepSimulation()
        time.sleep(1 / 500)

    link_pos_previous_9 = (0, 0, 0)
    foot_position_record = []

    output = []

    def get_joint_torques(robot_id, joint_indices):
        """
        Get the torques of the specified joints of the robot.

        Args:
            robot_id (int): The ID of the robot.
            joint_indices (list): The indices of the joints to get the torques.

        Returns:
            dict: A dictionary containing the joint indices as keys and the torques as values.
        """
        joint_torques = {}
        for joint_id in joint_indices:
            # Get the joint state
            joint_state = p.getJointState(robot_id, joint_id)
            torque = joint_state[3]
            joint_torques[joint_id] = torque
        return joint_torques

    def detect_contact(robot_id, link1, link2, plane_id):
        """
        Detect the contact information between the two links of the robot and the plane.
        """
        contact_points = p.getContactPoints(bodyA=robot_id, bodyB=plane_id)

        # Initialize the result
        result = {}
        for link in [link1, link2]:
            result[f"normal_force_{link}"] = 0.0
            result[f"lateral_friction_{link}"] = 0.0

        for point in contact_points:
            # check if the contact point is between the two links
            if point[3] == link1 or point[4] == link1:
                link_idx = link1
            elif point[3] == link2 or point[4] == link2:
                link_idx = link2
            else:
                continue

            # Get the normal force and lateral friction
            normal_force = point[9]
            lateral_friction = point[10]

            # Store the result
            result[f"normal_force_{link_idx}"] = normal_force
            result[f"lateral_friction_{link_idx}"] = lateral_friction

        return result

    ############################### Gait and transition locomotion ###############################
    network = ProcessNetwork()
    # 1. Load the CPG model data
    cpg_data = cpg_data[2500:, :]
    friction = []
    predicted_angles = []
    joint_torques = []

    if "_to_" in gait_type:
        before_gait, after_gait = gait_type.split("_to_")

    for i in range(len(cpg_data)):
        cpg_state = cpg_data[i, :]
        # 2. Process the data to the required format
        if "_to_" in gait_type:
            current_gait_type = before_gait if i < execution_time * 500 else after_gait
        else:
            current_gait_type = gait_type

        processed_data = network._process_data(cpg_state, current_gait_type)

        # 3. Predict the joint angles
        joint_angles = network._predict_joint_angles(processed_data)
        predicted_angles.append(joint_angles)

        # 4. Set the joint angles
        for j in range(12):
            targetPos = float(joint_angles[j])
            p.setJointMotorControl2(
                quadruped,
                jointIds[j],
                p.POSITION_CONTROL,
                jointDirections[j] * targetPos + jointOffsets[j],
                force=maxForceId[j],
            )

        # 5. Step the simulation
        p.stepSimulation()

        # Detect the contact information
        joint_torques.append(get_joint_torques(quadruped, jointIds))

        # 6. Record the data
        link_pos_current_9 = p.getLinkState(quadruped, 13)[0]

        # p.addUserDebugLine(
        #     link_pos_previous_9,
        #     link_pos_current_9,
        #     lineColorRGB=[1, 0, 0],
        #     lifeTime=10,
        #     lineWidth=2,
        # )

        link_pos_previous_9 = link_pos_current_9

        # Detect the contact information
        contact_info = detect_contact(quadruped, 12, 15, plane)

        # Get the normal force and lateral friction
        normal_force_12 = contact_info.get("normal_force_12", 0.0)
        lateral_friction_12 = contact_info.get("lateral_friction_12", 0.0)
        normal_force_15 = contact_info.get("normal_force_15", 0.0)
        lateral_friction_15 = contact_info.get("lateral_friction_15", 0.0)

        friction.append(
            [normal_force_12, lateral_friction_12, normal_force_15, lateral_friction_15]
        )

        foot_position_record.append(link_pos_current_9)
        base_linear_velocity, base_angular_velocity = p.getBaseVelocity(quadruped)
        current_time = [i / 500]
        body_pos = list(p.getBasePositionAndOrientation(quadruped)[0])
        euler = list(
            p.getEulerFromQuaternion(p.getBasePositionAndOrientation(quadruped)[1])
        )
        joint_pos = [p.getJointStates(quadruped, range(12))[i][0] for i in range(12)]
        output.append(
            current_time + body_pos + euler + joint_pos + list(link_pos_current_9)
        )
        time.sleep(1.0 / (1000.0))

    df = pandas.DataFrame(output, index=None)

    joint_file = (
        BLENDER_DATA_DIR / "gait_and_transition" / f"To_blender_{gait_type}_data.csv"
    )

    # Get the directory path
    directory = joint_file.parent
    # Check if the directory exists, if not, create it
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        joint_file,
        index=False,
        header=BLENDER_HEADERS
        + [
            "foot_traj_x",
            "foot_traj_y",
            "foot_traj_z",
        ],
    )
    p.disconnect()
    print("***CURRENT SIMULATION FINISHED***")


if __name__ == "__main__":
    # if gait_type = "walk", the robot will walk
    gait("walk", total_time=20, use_gui=True)
    # gait("trot", total_time=20, use_gui=True)
    # gait("pace", total_time=20, use_gui=True)
    # gait("bound", total_time=20, use_gui=True)
    # gait("pronk", total_time=20, use_gui=True)
