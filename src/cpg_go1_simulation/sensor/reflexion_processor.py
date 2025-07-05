from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from cpg_go1_simulation.stein.implementations import CPG8Neuron


@dataclass
class ReflexionConfig:
    """Configurations for the reflexion processor"""

    # Thresholds for slope detection
    UPHILL_THRESHOLD: float = 5.0
    DOWNHILL_THRESHOLD: float = -0.5

    # Size of the sliding window
    WINDOW_SIZE: int = 500

    # Continuous detection count required for gait switching
    GAIT_CHANGE_THRESHOLD: int = 20

    # Gait mapping
    GAIT_MAP = {"walk": 1, "trot": 2, "pace": 3, "bound": 4, "pronk": 5}


class ReflexionProcessor:
    def __init__(self, initial_gait: str = "trot", time_mask: Optional[float] = None):
        self.config = ReflexionConfig()
        self.last_gait = None  # Last executed gait
        self.current_gait = initial_gait  # Current executed gait
        self.last_detected_gait = initial_gait  # Last detected target gait
        self.torque_window = deque(maxlen=self.config.WINDOW_SIZE)

        # Gait change counter
        self.gait_change_counter = 0
        self.pending_gait = None

        # Record the time of the last execution
        self.execution_time = 0.0

        # Initialize the CPG as trot
        self.cpg = CPG8Neuron(
            before_ftype=self.config.GAIT_MAP["trot"],
            after_ftype=self.config.GAIT_MAP["trot"],
            total_time=0.002,
            toc=10.0,
        )

        # Initialize the wait switch flag
        self.wait_switch_flag = False

        # Initialize CPG state
        # If a time mask exists, initialize CPG with the mask
        if time_mask is not None:
            self.time_mask = time_mask
            self.last_state = self.cpg.calculate_start_data(self.time_mask)
        else:
            self.time_mask = 0.0
            self.last_state = self.cpg.pos0

    def update_target_gait(self, avg_torque: float):
        neuron_14 = 1 / (1 + np.exp(-(avg_torque - self.config.UPHILL_THRESHOLD)))
        neuron_15 = 1 / (1 + np.exp(avg_torque - self.config.DOWNHILL_THRESHOLD))
        neuron_16 = round(neuron_14) ^ round(neuron_15)

        if neuron_16:
            target_gait = "walk"
        else:
            target_gait = "trot"
        return target_gait

    def process_torque_info(self, sum_torque: float) -> Tuple[str, float]:
        """Process the torque information and determine the target gait
        Args:
            sum_torque (float): The total torque to process.
        Returns:
            Tuple[str, float]: The detected gait and the average torque.
        """
        self.torque_window.append(sum_torque)

        if len(self.torque_window) < self.config.WINDOW_SIZE:
            return self.last_detected_gait, 0.0

        avg_torque = np.mean(self.torque_window)

        # Determine the target gait
        target_gait = self.update_target_gait(avg_torque)

        # Update the gait change counter
        if target_gait != self.last_detected_gait:
            if target_gait == self.pending_gait:
                self.gait_change_counter += 1
                print(
                    f"Gait change count: {self.gait_change_counter}/{self.config.GAIT_CHANGE_THRESHOLD}"
                )
            else:
                self.gait_change_counter = 1
                self.pending_gait = target_gait
        else:
            self.gait_change_counter = 0
            self.pending_gait = None

        # Only update the detected gait when the counter reaches the threshold
        if self.gait_change_counter >= self.config.GAIT_CHANGE_THRESHOLD:
            new_gait = target_gait
            self.gait_change_counter = 0
            self.pending_gait = None
            self.last_detected_gait = new_gait
        else:
            new_gait = self.last_detected_gait

        return new_gait, avg_torque

    def update(
        self, sim_time: float, sum_torque: float
    ) -> Tuple[str, np.ndarray, float]:
        """Update the reflexion processor with the current torque information.
        Args:
            sim_time (float): The current simulation time.
            sum_torque (float): The total torque to process.
        Returns:
            Tuple[str, np.ndarray, float]: The detected gait, the CPG output, and the average torque.
        """
        detected_gait, avg_torque = self.process_torque_info(sum_torque)
        gait_changed = detected_gait != self.current_gait

        # Check if the gait has changed
        if gait_changed and sim_time > 0.5 and (sim_time - self.execution_time) > 0.5:
            # Check if the detected gait is different from the current gait
            if self.current_gait == "walk" and detected_gait == "trot":
                if not self.wait_switch_flag:
                    # Set the wait switch flag
                    self.wait_switch_flag = True
                    # Update the CPG to switch from walk to trot, and continue the current CPG until the switch is complete
                    self.cpg = CPG8Neuron(
                        before_ftype=self.config.GAIT_MAP["walk"],
                        after_ftype=self.config.GAIT_MAP["trot"],
                        total_time=0.002,
                        toc=sim_time + self.time_mask,
                        counter=self.cpg.counter,
                    )
                    return self.current_gait, self.calculate_cpg(sim_time), avg_torque

                # Check if the CPG has reached the target gait
                if hasattr(self.cpg, "wait_toc") and self.cpg.wait_toc > 0:
                    # if true, means the CPG has reached the target gait

                    self.wait_switch_flag = False
                    self.current_gait = detected_gait
                    self.execution_time = sim_time
                    return self.current_gait, self.calculate_cpg(sim_time), avg_torque
                else:
                    # If not met, continue waiting
                    return self.current_gait, self.calculate_cpg(sim_time), avg_torque
            else:
                # Directly switch from trot to walk
                self.cpg = CPG8Neuron(
                    before_ftype=self.config.GAIT_MAP["trot"],
                    after_ftype=self.config.GAIT_MAP["walk"],
                    total_time=0.002,
                    toc=sim_time + self.time_mask,
                    counter=self.cpg.counter,
                )
                self.current_gait = detected_gait
                self.execution_time = sim_time  # Record the switch time
                # Reset the wait switch flag
                self.wait_switch_flag = False
                print(
                    f"Directly switching gait: {self.current_gait} -> {detected_gait}"
                )

        return self.current_gait, self.calculate_cpg(sim_time), avg_torque

    def calculate_cpg(self, current_time: float) -> np.ndarray:
        """Calculate the CPG output at the current time."""
        current_data, current_state = self.cpg.calculate_current_data(
            last_state=self.last_state, current_time=current_time + self.time_mask
        )
        self.last_state = current_state
        return current_data.reshape(1, -1)
