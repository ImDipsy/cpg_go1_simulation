from pathlib import Path

from cpg_go1_simulation.sensor import MODEL_PATH

# ROOT DIRECTORY
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Gait mapping
GAIT_MAP = {"walk": 1, "trot": 2, "pace": 3, "bound": 4, "pronk": 5}

# DATA DIRECTORY
DATA_DIR = ROOT_DIR / "data"
CPG_DATA_DIR = DATA_DIR / "cpg_data"
GAIT_DATA_DIR = CPG_DATA_DIR / "gait_data"
BACKWARD_DATA_DIR = CPG_DATA_DIR / "backward_data"
TRANSITION_DATA_DIR = CPG_DATA_DIR / "transition_data"
JOINT_DATA_PATH = DATA_DIR / "joint_control_data"

MLR_PERTURBATION_DATA_DIR = CPG_DATA_DIR / "perturbation_data" / "mlr"
STATE_PERTURBATION_DATA_DIR = CPG_DATA_DIR / "perturbation_data" / "cpg_state"
# SUCCESS RATE DATA
CPG_SUCCESS_RATE_DIR = DATA_DIR / "cpg_success_rate"
# BACKWARD ANIMATION PATH
BACKWARD_ANIMATIONS_DIR = DATA_DIR / "backward_locomotion" / "video_creating_data"


TRAINING_DIR = DATA_DIR / "Training"
TRAIN_DATA_DIR = TRAINING_DIR / "train_data"
TRAIN_MODEL_DIR = TRAINING_DIR / "train_model"
TRAIN_RESULT_DIR = TRAINING_DIR / "train_result"

# ROBOT MODEL PATH
ROBOT_MODEL_DIR = MODEL_PATH
GO1_MODEL_DIR = ROBOT_MODEL_DIR / "go1"
TERRAIN_DIR = ROBOT_MODEL_DIR / "terrain"

# MODEL PATH
BEST_MODEL_PATH = ROOT_DIR / "resources" / "best_model" / "best_model_v1.2.pt"

# SENSOR OUTPUT DATA
OUTPUT_DATA_DIR = DATA_DIR / "Output"
# BLENDER DATA
BLENDER_DATA_DIR = OUTPUT_DATA_DIR / "video_creating_data"
# TRAJECTORY PLOT DATA
TRAJECTORY_PLOT_DIR = OUTPUT_DATA_DIR / "trajectory_plots"
# SENSOR OUTPUT IMAGE
OUTPUT_IMAGE_DIR = OUTPUT_DATA_DIR / "images"
# CONTROL DECISION DATA
CONTROL_DATA_DIR = OUTPUT_DATA_DIR / "control_data"
# CONTACT DATA
CONTACT_DATA_DIR = OUTPUT_DATA_DIR / "contact_data"

# BLENDER DATA Headers
BLENDER_HEADERS = [
    "time",
    "body_x	",
    "body_y",
    "body_z",
    "body_rx",
    "body_ry",
    "body_rz",
    "RL_hip_rx_1",
    "RL_thigh_ry_1",
    "RL_calf_ry_1",
    "RR_hip_rx_2",
    "RR_thigh_ry_2",
    "RR_calf_ry_2",
    "FR_hip_rx_3",
    "FR_thigh_ry_3",
    "FR_calf_ry_3",
    "FL_hip_rx_4",
    "FL_thigh_ry_4",
    "FL_calf_ry_4",
]

TORQUES_HEADERS = [
    "time",
    "body_x	",
    "body_y",
    "body_z",
    "body_rx",
    "body_ry",
    "body_rz",
    "RL_hip_torque",
    "RR_hip_torque",
    "FR_hip_torque",
    "FL_hip_torque",
    "sum_torque",
    "avg_torque",
    "current_gait",
]

# CPG PERTURBATION DATA Headers
CPG_MLR_PERTURBATION_HEADERS = [
    "time",
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6",
    "x7",
    "x8",
    "a_hip",
    "f_hip",
    "k1_hip",
    "k2_hip",
    "a_knee",
    "f_knee",
    "k1_hip",
    "k2_hip",
]

# CPG STATE PERTURBATION DATA Headers
CPG_STATE_PERTURBATION_HEADERS = [
    "time",
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6",
    "x7",
    "x8",
]
