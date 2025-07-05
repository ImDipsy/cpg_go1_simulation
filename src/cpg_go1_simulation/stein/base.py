# coding:utf-8
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas


@dataclass
class Args:
    ftype: list[int]
    total_time: float
    neuron_type: str = (
        "8neuron"  # "8neuron", "10neuron", "12neuron", "8neuron_backward"
    )
    toc: float = 10.0
    _if_backward: bool = False
    _if_mlr_perturbation: bool = False
    _if_state_perturbation: bool = False


class CPGBase(ABC):
    """Base class for CPG implementations"""

    def __init__(
        self, before_ftype: int, after_ftype: int, total_time: float, toc: float
    ):
        """
        Common initialization for all CPG types
        """
        # Basic parameters
        self.before_ftype = before_ftype
        self.after_ftype = after_ftype
        self.toc = toc
        self.time_step = 1 / 10000
        self.time_freq = int(1 / self.time_step)
        self.counter = 0
        self.tspan = [0, total_time]
        self.gaitnames = {1: "walk", 2: "trot", 3: "pace", 4: "bound", 5: "pronk"}

    @abstractmethod
    def set_link_params(self, ftype: int):
        """Set link parameters specific to each implementation"""
        pass

    @abstractmethod
    def set_pos0(self):
        """Set initial positions specific to each implementation"""
        pass

    @abstractmethod
    def set_driving_signal(self):
        """Set driving signals specific to each implementation"""
        pass

    @abstractmethod
    def set_strategy(self):
        """Set control strategy for different network architectures"""
        pass

    @abstractmethod
    def get_output_path(self):
        """Get output file path specific to each implementation"""
        pass

    @abstractmethod
    def export_data(self):
        """export data"""
        pass

    @abstractmethod
    def export_csv(self):
        """Export data to csv file"""
        pass

    @abstractmethod
    def stein(self, pos: np.ndarray, t: float):
        """Stein oscillator implementation specific to each type"""
        # counter for gait transition
        pass

    def get_neuron_params(self, ftype: int) -> None:
        """Common neuron parameters for all implementations"""
        # justify which mode is
        if ftype == 1:
            # walk
            # hip layer
            a_hip, f_hip, k1_hip, k2_hip = 10, 40, 0, 0
            # knee layer
            a_knee, f_knee, k1_knee, k2_knee = 10, 40, 0, 0
        elif ftype == 2:
            # trot
            # hip layer
            a_hip, f_hip, k1_hip, k2_hip = 11, 41, 0.085, 56
            a_knee, f_knee, k1_knee, k2_knee = 11, 41, 0, 0
        elif ftype == 3:
            # pace
            # hip layer
            a_hip, f_hip, k1_hip, k2_hip = 11, 41, 0.04, 54
            a_knee, f_knee, k1_knee, k2_knee = 11, 41, 0.01, 54
        elif ftype == 4:
            # bound
            # hip layer
            a_hip, f_hip, k1_hip, k2_hip = 16, 50, 0.1, 59
            a_knee, f_knee, k1_knee, k2_knee = 14, 45, 0, 0
        elif ftype == 5:
            a_hip, f_hip, k1_hip, k2_hip = 22, 65, 0.3, 60
            a_knee, f_knee, k1_knee, k2_knee = 22, 65, 0.2, 60

        return a_hip, f_hip, k1_hip, k2_hip, a_knee, f_knee, k1_knee, k2_knee

    def set_neuron_params(self, ftype: int) -> None:
        """Common neuron parameter setting"""
        self.b_param = -2000
        self.p_param = 10
        self.q_param = 30

        (
            self.a_hip,
            self.f_hip,
            self.k1_hip,
            self.k2_hip,
            self.a_knee,
            self.f_knee,
            self.k1_knee,
            self.k2_knee,
        ) = self.get_neuron_params(ftype)

    def runge_kutta(self, pos: np.ndarray, t: float) -> np.ndarray:
        """
        fourth order runge_kutta method for stein oscillator
        """

        # stein function
        data = []
        t0 = t[0]  # initial time
        h = self.time_step
        h_half = self.time_step / 2
        x0 = pos

        for j in range(len(t)):
            K1 = self.stein(x0, t0)
            K2 = self.stein(x0 + h * K1 / 2, t0 + h_half)
            K3 = self.stein(x0 + h * K2 / 2, t0 + h_half)
            K4 = self.stein(x0 + h * K3, t0 + h)
            x1 = x0 + (h / 6) * (K1 + 2 * K2 + 2 * K3 + K4)

            data.append(list(x1))

            # update x0 and t0
            x0 = x1
            t0 = self.decimal(t0 + h, 5)
        return np.array(data)

    def decimal(self, data: float, digits: int) -> float:
        """Set siginificant digits for data"""
        return round(data, digits)

    def save_data(
        self,
        file_path: str,
        data: np.ndarray,
        headers: Optional[list[str]] = None,
        index: list[str] = False,
    ) -> None:
        # 获取目录路径
        directory = file_path.parent
        # 检查目录是否存在，如果不存在则创建
        directory.mkdir(parents=True, exist_ok=True)
        df = pandas.DataFrame(data)
        df.to_csv(file_path, header=headers, index=index)
        print(f"Exported to {file_path}")
