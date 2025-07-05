import math
from typing import Optional

import numpy as np

from cpg_go1_simulation.config import (
    BACKWARD_DATA_DIR,
    CPG_MLR_PERTURBATION_HEADERS,
    CPG_STATE_PERTURBATION_HEADERS,
    GAIT_DATA_DIR,
    MLR_PERTURBATION_DATA_DIR,
    STATE_PERTURBATION_DATA_DIR,
    TRANSITION_DATA_DIR,
)
from cpg_go1_simulation.stein.base import CPGBase


class CPG8Neuron(CPGBase):
    """8 neuron implementation for both forward and backward locomotion."""

    def __init__(
        self,
        before_ftype: int,
        after_ftype: int,
        total_time: float,
        toc: float,
        _if_backward: bool = False,
        _if_mlr_perturbation: bool = False,
        _if_state_perturbation: bool = False,
        counter: Optional[int] = None,
        f_point: Optional[int] = None,
    ):
        super().__init__(before_ftype, after_ftype, total_time, toc)
        self._if_backward = _if_backward
        self._if_mlr_perturbation = _if_mlr_perturbation
        self._if_state_perturbation = _if_state_perturbation
        self.noise_freq = 100

        if self._if_backward:
            init_gamma, init_delta = -0.1, -0.6
        else:
            init_gamma, init_delta = -0.6, -0.1

        self.neuron_nums = 8
        self.fci_list = []
        self.real_toc = 0
        self.control_time_step = 0.002
        self.counter = counter if counter else 0
        self.point = f_point if f_point else 0

        # Network matrix
        self.network_matrix = [
            [-0.15, -0.15, init_gamma, init_delta],
            [-0.15, -0.15, -0.6, -0.1],
            [-0.15, -0.15, -0.6, -0.1],
            [-0.15, -0.15, -0.6, -0.1],
            [-0.15, -0.15, -0.6, -0.1],
        ]

        # Initialize parameters
        self.set_neuron_params(ftype=self.before_ftype)
        self.set_link_params(ftype=self.before_ftype)
        self.set_pos0()
        self.set_driving_signal()

        # Set strategy
        self.set_strategy()

        if self._if_mlr_perturbation:
            self.neuron_params_list = []

    def get_init_neuron_params(self, ftype):
        self.neuron_params = self.get_neuron_params(ftype)

    def set_mlr_perturbation_params(self, ftype: int, percentage: float) -> None:
        """Initialize parameters for MLR perturbation
        Args:
            ftype: gait type
            percentage: perturbation percentage, unit: %
        """

        # Set neuron parameters with MLR perturbation
        # neuron_params = self.get_neuron_params(ftype)
        neuron_params = self.neuron_params

        # Set Uniform Noise
        noise_range = [param * percentage / 100 for param in neuron_params]
        noise = [np.random.uniform(-val, val) for val in noise_range]
        perturbed_params = [
            param + noise_val for param, noise_val in zip(neuron_params, noise)
        ]
        # perturbed_params = [param * (1 + percentage / 100) for param in neuron_params]

        [
            self.a_hip,
            self.f_hip,
            self.k1_hip,
            self.k2_hip,
            self.a_knee,
            self.f_knee,
            self.k1_knee,
            self.k2_knee,
        ] = perturbed_params

        self.set_driving_signal()

    def set_strategy(self) -> None:
        # strategy
        if self.before_ftype == self.after_ftype:
            self.strategy = 1  # no gait transition, default strategy

        else:
            # Switch strategy
            if [self.before_ftype, self.after_ftype] in [
                [1, 4],
                [1, 5],
                [2, 1],
                [2, 4],
                [2, 5],
                [3, 1],
                [3, 4],
                [3, 5],
                [4, 5],
            ]:
                self.strategy = 1

            # Power Pair strategy
            elif [self.before_ftype, self.after_ftype] in [[5, 1], [5, 4]]:
                self.strategy = 2
                # initialize Power Pair strategy
                self.init_powerpair_strategy()

            # Wait&Switch strategy
            elif [self.before_ftype, self.after_ftype] in [
                [1, 2],
                [1, 3],
                [2, 3],
                [3, 2],
            ]:
                self.strategy = 3
                # add Wait&Switch flag
                self.wait_switch_called = False
                self.wait_toc = 0

            # Wait&Power Pair strategy
            elif [self.before_ftype, self.after_ftype] in [
                [4, 1],
                [4, 2],
                [4, 3],
                [5, 2],
                [5, 3],
            ]:
                self.strategy = 4
                # initialize Power Pair strategy
                self.init_powerpair_strategy()
                # add Wait&Power Pair flag
                self.wait_powerpair_called = False
                self.wait_toc = 0

    def set_link_params(self, ftype: int) -> None:
        """Initialize link parameters for 8 neuron network"""

        # Set link parameters
        [self.alpha, self.beta, self.gamma, self.delta] = [
            self.network_matrix[ftype - 1][i] for i in range(4)
        ]

        # Initialize coupling matrices
        lambda_1 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
        lambda_2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]])

        # Global coordinates
        global_layer1_num = np.array([1, 2, 3, 4])
        global_layer2_num = np.array([5, 6, 7, 8])

        # Initialize coupling matrices
        self.link_omega = np.zeros((self.neuron_nums, self.neuron_nums))
        self.link_kappa_1and2 = np.zeros((self.neuron_nums, self.neuron_nums))

        # Set coupling values
        for i in range(4):
            for j in range(4):
                self.link_omega[global_layer1_num[i] - 1, global_layer1_num[j] - 1] = (
                    lambda_1[i, j]
                )
                self.link_omega[global_layer2_num[i] - 1, global_layer2_num[j] - 1] = (
                    lambda_2[i, j]
                )
                self.link_kappa_1and2[
                    global_layer2_num[i] - 1, global_layer1_num[i] - 1
                ] = 1
                self.link_kappa_1and2[
                    global_layer1_num[i] - 1, global_layer2_num[i] - 1
                ] = 1

    def set_pos0(self) -> None:
        """Initialize initial positions for 8 neuron network"""
        pos0 = np.array(
            [
                [1, 0.04, 0.016],
                [1, 0.045, 0.018],
                [0.8, 0.05, 0.02],
                [1, 0.025, 0.014],
                [1, 0.045, 0.018],
                [0.8, 0.05, 0.02],
                [1, 0.025, 0.014],
                [1, 0.04, 0.016],
            ]
        )
        self.pos0 = np.hstack((pos0[:, 0], pos0[:, 1], pos0[:, 2]))

    def set_driving_signal(self) -> None:
        """Set driving signals for 8 neuron network"""
        self.VECTOR_f = np.hstack(
            np.array([self.f_hip * np.ones(4), self.f_knee * np.ones(4)])
        )
        self.VECTOR_k1 = np.hstack(
            np.array([self.k1_hip * np.ones(4), self.k1_knee * np.ones(4)])
        )
        self.VECTOR_k2 = np.hstack(
            np.array([self.k2_hip * np.ones(4), self.k2_knee * np.ones(4)])
        )
        self.VECTOR_a = np.hstack(
            np.array([self.a_hip * np.ones(4), self.a_knee * np.ones(4)])
        )
        self.VECTOR_omega = np.hstack(
            np.array([self.alpha * np.ones(4), self.beta * np.ones(4)])
        )
        self.VECTOR_kappa_1 = np.hstack(
            np.array([self.gamma * np.ones(4), self.delta * np.ones(4)])
        )

        self.a_all = np.diag(self.VECTOR_a)
        self.f_all = np.diag(self.VECTOR_f)
        self.k1_all = np.diag(self.VECTOR_k1)
        self.k2_all = self.VECTOR_k2

        self.combined_coup_matrix_with_weight = (
            self.VECTOR_omega * self.link_omega
            + self.VECTOR_kappa_1 * self.link_kappa_1and2
        )

    def init_powerpair_strategy(self) -> None:
        """
        The following variables define the time transition gap (t_trans_gap) and the amplification rate (rate) for the Power Pair strategy.
        - t_trans_gap: Time duration for the transition gap in the Power Pair strategy.
        - rate: Amplification factor for the Power Pair strategy.

        Note: Make sure that int(self.t_trans_gap * self.ratio_pre * self.time_freq) is an integer to avoid errors.
        """
        if [self.before_ftype, self.after_ftype] == [5, 1]:
            [t_trans_gap, rate, ratio_pre, ratio_after] = [0.1, 2, 0.1, 0.9]
        elif [self.before_ftype, self.after_ftype] == [5, 2]:
            [t_trans_gap, rate, ratio_pre, ratio_after] = [0.07, 2.6, 0.1, 0.9]
        elif [self.before_ftype, self.after_ftype] == [5, 3]:
            [t_trans_gap, rate, ratio_pre, ratio_after] = [0.09, 2.6, 0.1, 0.9]
        elif [self.before_ftype, self.after_ftype] == [5, 4]:
            [t_trans_gap, rate, ratio_pre, ratio_after] = [0.14, 2, 0.1, 0.9]
        elif [self.before_ftype, self.after_ftype] == [4, 1]:
            [t_trans_gap, rate, ratio_pre, ratio_after] = [0.2, 2.2, 0.1, 0.9]
        elif [self.before_ftype, self.after_ftype] == [4, 2]:
            [t_trans_gap, rate, ratio_pre, ratio_after] = [0.4, 1.8, 0.4, 0.6]
        elif [self.before_ftype, self.after_ftype] == [4, 3]:
            [t_trans_gap, rate, ratio_pre, ratio_after] = [0.2, 2, 0.1, 0.9]
        # time interval for gait transition
        self.t_trans_gap = t_trans_gap
        # amplification factor for f values
        self.rate = rate
        self.ratio_pre = ratio_pre
        self.ratio_after = ratio_after
        # time when the transition period ends
        self.t_trans_end = self.decimal(self.toc + self.t_trans_gap, 8)
        # obtain f_hip values before and after the transition period
        self.f_hip_pre = self.get_neuron_params(ftype=self.before_ftype)[1]
        self.f_hip_after = self.get_neuron_params(ftype=self.after_ftype)[1]
        # counter to track the number of power pair strategy invocations

        self.cal_f()

    def cal_f(self) -> None:
        """
        calculate the logarithmic change in f values and scale it between the corresponding f values.
        """
        num_pre = int(self.t_trans_gap * int(self.ratio_pre * self.time_freq)) + 2
        y_pre = [math.log(i, 10) for i in range(1, num_pre)]
        self.f_powerpair_pre = [
            y * ((self.rate * self.f_hip_pre - self.f_hip_pre) / (y_pre[-1] - y_pre[0]))
            + self.f_hip_pre
            for y in y_pre
        ]

        num_after = (
            int(self.t_trans_gap * self.time_freq)
            - int(self.ratio_after * self.t_trans_gap * self.time_freq)
            + 2
        )
        y_after = [math.log(i, 0.1) for i in range(1, num_after)]
        self.f_powerpair_after = [
            self.rate * self.f_hip_pre
            + y
            * (
                (self.rate * self.f_hip_pre - self.f_hip_after)
                / (y_after[0] - y_after[-1])
            )
            for y in y_after
        ]

    def append_f(self, f: float) -> None:
        """
        record the fci values under the power pair strategy
        """
        self.fci_list.append(f)

    def set_powerpair_strategy(self, t: float, real_toc: float) -> None:
        """
        set parameters of powerpair strategy
        """

        # apply logarithmic excitation signal during the first ratio_pre of the transition period.
        if t <= real_toc + self.t_trans_gap * self.ratio_pre:
            self.f_hip_trans = self.f_powerpair_pre[self.point]

        # keep f value unchanged during the ratio_pre to ratio_after of the transition period.

        # apply exponential excitation signal to transition to f during the ratio_after to 100% of the transition period.
        elif (
            t >= real_toc + self.t_trans_gap * self.ratio_after
            and t <= real_toc + self.t_trans_gap
        ):
            self.f_hip_trans = self.f_powerpair_after[
                self.point - int(self.t_trans_gap * self.ratio_after * self.time_freq)
            ]

        if [self.before_ftype, self.after_ftype] in [
            [1, 2],
            [3, 2],
            [4, 2],
            [5, 2],
            [4, 1],
            [5, 1],
        ]:
            f_hip_list = np.hstack(
                np.array(
                    [self.f_hip_trans, self.f_hip_pre, self.f_hip_trans, self.f_hip_pre]
                )
            )

        elif [self.before_ftype, self.after_ftype] in [
            [1, 3],
            [2, 3],
            [4, 3],
            [5, 3],
        ]:
            f_hip_list = np.hstack(
                np.array(
                    [
                        self.f_hip_trans,
                        self.f_hip_pre,
                        self.f_hip_pre,
                        self.f_hip_trans,
                    ]
                )
            )

        elif [self.before_ftype, self.after_ftype] in [[5, 4]]:
            f_hip_list = np.hstack(
                np.array(
                    [self.f_hip_trans, self.f_hip_trans, self.f_hip_pre, self.f_hip_pre]
                )
            )

        # update vector f
        self.VECTOR_f = np.hstack(
            np.array([f_hip_list * np.ones(4), self.f_knee * np.ones(4)])
        )
        self.f_all = np.diag(self.VECTOR_f)

    def set_wait_switch(self, t0: float, x0: float, x1: float) -> None:
        """
        set parameters of wait&switch strategy
        """
        if [self.before_ftype, self.after_ftype] == [1, 2]:
            if x1 < x0 and 0.3 < x1 < 0.48:
                self.set_gait_trans_params()
                self.wait_switch_called = True
                self.wait_toc = t0
                # print("set gait transition parameters", t0)
            else:
                pass
        elif [self.before_ftype, self.after_ftype] == [1, 3]:
            if x1 > x0 and 0.3 < x1 < 0.560:
                self.set_gait_trans_params()
                self.wait_switch_called = True
                self.wait_toc = t0
            else:
                pass
        elif [self.before_ftype, self.after_ftype] == [2, 3]:
            if x1 < x0 and 0.16 < x1 < 0.19:
                self.set_gait_trans_params()
                self.wait_switch_called = True
                self.wait_toc = t0
            else:
                pass
        elif [self.before_ftype, self.after_ftype] == [3, 2]:
            if x1 < x0 and 0.385 < x1 < 0.502:
                self.set_gait_trans_params()
                self.wait_switch_called = True
                self.wait_toc = t0
            else:
                pass
        else:
            print("Gait transition type does not match wait&switch strategy.")

    def set_wait_powerpair(self, t0: float, x0: float, x1: float) -> None:
        """
        set parameters of wait&power pair strategy
        """
        if [self.before_ftype, self.after_ftype] == [4, 1]:
            if (x1 < x0 and x1 > 0.307) or (x1 > x0 and x1 > 0.56):
                self.wait_powerpair_called = True
                self.wait_toc = t0

        elif [self.before_ftype, self.after_ftype] == [4, 2]:
            if (x1 < x0 and x1 > 0.625) or (x1 > x0 and x1 > 0.580):
                self.wait_powerpair_called = True
                self.wait_toc = t0

        elif [self.before_ftype, self.after_ftype] == [4, 3]:
            if x1 < x0 and 0.149 < x1 < 0.180:
                self.wait_powerpair_called = True
                self.wait_toc = t0

        elif [self.before_ftype, self.after_ftype] == [5, 2]:
            if x1 > x0 and 0.520 < x1 < 0.700:
                self.wait_powerpair_called = True
                self.wait_toc = t0

        elif [self.before_ftype, self.after_ftype] == [5, 3]:
            if x1 < x0 and x1 < 0.250:
                self.wait_powerpair_called = True
                self.wait_toc = t0

    def set_gait_trans_params(self) -> None:
        """
        set parameters after gait transition
        """
        self.set_neuron_params(self.after_ftype)
        # set link parameters after gait transition
        self.set_link_params(self.after_ftype)
        # set driving signal after gait transition
        self.set_driving_signal()

    def set_gait_transition(self, t0: float, x0: float, x1: float) -> None:
        """
        set up gait transition for each strategy
        """
        if self.strategy == 1:
            if t0 == self.toc:
                self.set_gait_trans_params()

        elif self.strategy == 2:
            if t0 >= self.toc and t0 <= self.decimal(self.toc + self.t_trans_gap, 5):
                self.set_powerpair_strategy(t0, self.toc)
                self.point += 1

                if t0 == self.decimal(self.toc + self.t_trans_gap, 5):
                    self.set_gait_trans_params()
                    # print("set gait transition parameters", t0)

        # Wait&Switch strategy
        elif self.strategy == 3:
            if t0 >= self.toc and not self.wait_switch_called:
                self.set_wait_switch(t0, x0[0], x1[0])

        # Wait&Power Pair strategy
        elif self.strategy == 4:
            if t0 >= self.toc and not self.wait_powerpair_called:
                self.set_wait_powerpair(t0, x0[0], x1[0])

            if self.wait_powerpair_called:
                if t0 >= self.wait_toc and t0 <= self.decimal(
                    self.wait_toc + self.t_trans_gap, 5
                ):
                    self.set_powerpair_strategy(t0, self.wait_toc)
                    self.point += 1

                    if t0 == self.decimal(self.wait_toc + self.t_trans_gap, 5):
                        self.set_gait_trans_params()
                        # print("set gait transition parameters", t0)

    def set_stimulate_time(self, current_time: float) -> float:
        """set stimulate time according to different gait strategys"""
        if self.before_ftype != self.after_ftype:
            if self.strategy == 1:
                if self.counter >= int(self.toc * int(4 * self.time_freq)) + 1:
                    t_stimulate = current_time - self.toc
                else:
                    t_stimulate = current_time
            elif self.strategy == 2:
                if self.counter >= int(self.t_trans_end * int(4 * self.time_freq)) + 1:
                    t_stimulate = current_time - self.t_trans_end
                else:
                    t_stimulate = current_time
            elif self.strategy == 3:
                if self.wait_toc > 0:
                    if self.counter >= int(self.wait_toc * int(4 * self.time_freq)) + 1:
                        t_stimulate = current_time - self.wait_toc
                else:
                    t_stimulate = current_time
            elif self.strategy == 4:
                if self.wait_toc > 0:
                    tend = self.decimal(self.wait_toc + self.t_trans_gap, 8)
                    if self.counter >= int(tend * int(4 * self.time_freq)) + 1:
                        t_stimulate = current_time - tend
                else:
                    t_stimulate = current_time
        else:
            t_stimulate = current_time

        return self.decimal(t_stimulate, 8)

    def stein(self, pos: np.ndarray, t: float) -> np.ndarray:
        """Stein oscillator implementation for 8 neuron network"""
        self.counter += 1

        x = pos[0 : self.neuron_nums]
        y = pos[self.neuron_nums : 2 * self.neuron_nums]
        z = pos[2 * self.neuron_nums :]

        t = self.decimal(t, 8)

        t_stimulate = self.set_stimulate_time(t)

        fci = self.f_all @ (
            1
            + self.k1_all @ np.sin(self.k2_all * t_stimulate)
            + self.combined_coup_matrix_with_weight @ x
        )

        dxdt = self.a_all @ (
            -x + 1 / (1 + np.exp(-fci - self.b_param * y + self.b_param * z))
        )
        dydt = x - self.p_param * y
        dzdt = x - self.q_param * z

        return np.hstack((dxdt, dydt, dzdt))

    def calculate_start_data(self, time_mask: float) -> tuple[np.ndarray, int]:
        """calculate the initial state of the gait transition"""
        time_list = np.round(
            np.linspace(0, time_mask, int(time_mask / self.time_step) + 1), 5
        )
        time_list = time_list[:-1]

        _, start_state = self.runge_kutta_x_and_xdot(self.pos0, time_list)
        return start_state

    def calculate_current_data(
        self, last_state: np.ndarray, current_time: float
    ) -> np.ndarray:
        """calculate the real time of the gait transition"""
        time = np.linspace(
            current_time,
            current_time + self.control_time_step - self.time_step,
            20,
        )
        data, current_state = self.runge_kutta_x_and_xdot(last_state, time)

        current_data = data[0, :]

        return current_data, current_state

    def runge_kutta_perturbation(self, pos: np.ndarray, t: list[float]) -> np.ndarray:
        # stein function
        data = []
        t0 = t[0]  # initial time
        h = self.time_step
        h_half = self.time_step / 2
        x0 = pos

        data.append(x0[0:8])

        for j in range(len(t)):
            K1 = self.stein(x0, t0)
            K2 = self.stein(x0 + h * K1 / 2, t0 + h_half)
            K3 = self.stein(x0 + h * K2 / 2, t0 + h_half)
            K4 = self.stein(x0 + h * K3, t0 + h)
            x1 = x0 + (h / 6) * (K1 + 2 * K2 + 2 * K3 + K4)

            data.append(list(x1[0:8]))

            # set the time points for gait transition or perturbation disturbance to take effect, both set to the next time step.
            # when j = 0, t = 0 s; when j = 100, t = time_step * j = 0.01 s
            # add perturbation disturbance, defaulting to take effect in the next time step
            if self._if_state_perturbation:
                # At time = 11s, apply a perturbation of 0.1 to the first four neurons
                if j == int(11 * self.time_freq):
                    x1[0:4] = x1[0:4] + 0.1

                # At time = 12s, apply a random perturbation in the range [-0.08, 0.08] to the first four neurons
                if j == int(12 * self.time_freq):
                    perturbation = np.random.rand(1, 4) * 0.16 - 0.08
                    x1[0:4] = x1[0:4] + perturbation

                # 在 13-14s 期间，施加随机噪声，范围为 [-0.008, 0.008]
                elif j >= int(13 * self.time_freq) and j < int(14 * self.time_freq):
                    if j % (self.time_freq // self.noise_freq) == 0:
                        perturbation = np.random.rand(1, 4) * 0.008 - 0.004
                        x1[0:4] = x1[0:4] + perturbation

                # 在 14-15s 期间，施加随机噪声，范围为 [-0.005, 0.005]
                elif j >= int(14 * self.time_freq) and j < int(15 * self.time_freq):
                    if j % (self.time_freq // self.noise_freq) == 0:
                        perturbation = np.random.rand(1, 4) * 0.01 - 0.005
                        x1[0:4] = x1[0:4] + perturbation

            if self._if_mlr_perturbation:
                if j >= int(11 * self.time_freq):
                    if j % (self.time_freq // self.noise_freq) == 0:
                        self.set_mlr_perturbation_params(
                            ftype=self.before_ftype, percentage=0.02
                        )

                self.neuron_params_list.append(
                    [
                        self.a_hip,
                        self.f_hip,
                        self.k1_hip,
                        self.k2_hip,
                        self.a_knee,
                        self.f_knee,
                        self.k1_knee,
                        self.k2_knee,
                    ]
                )

            # update x0 and t0
            x0 = x1
            t0 = t0 + h
            t0 = self.decimal(t0, 5)
        return np.array(data)

    def runge_kutta_x_and_xdot(self, pos: np.ndarray, t: list[float]) -> np.ndarray:
        """
        fourth order runge_kutta method for stein oscillator
        """

        # stein function
        data = []
        t0 = t[0]  # initial time
        h = self.time_step
        h_half = self.time_step / 2
        x0 = pos
        # data.append(x0[0:8])

        x_dot = []

        for j in range(len(t)):
            k1 = self.stein(x0, t0)
            k2 = self.stein(x0 + h * k1 / 2, t0 + h_half)
            k3 = self.stein(x0 + h * k2 / 2, t0 + h_half)
            k4 = self.stein(x0 + h * k3, t0 + h)
            x1 = x0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            x_dot = (x1 - x0) / h

            data.append(list(x1[0:8]) + list(x_dot[0:8]))
            current_state = x1

            # save fci values under the Power Pair and Wait&Power Pair strategy
            self.append_f(self.VECTOR_f)

            if self.before_ftype != self.after_ftype:
                self.set_gait_transition(t0, x0, x1)

            # update x0 and t0
            x0 = x1
            t0 = t0 + h
            t0 = self.decimal(t0, 5)
        return np.array(data), current_state

    def get_output_path(self) -> str:
        if self._if_mlr_perturbation and not self._if_state_perturbation:
            return (
                MLR_PERTURBATION_DATA_DIR
                / f"cpg_{self.gaitnames[self.before_ftype]}_perturbation_{self.tspan[1]}s.csv"
            )
        elif self._if_state_perturbation and not self._if_mlr_perturbation:
            return (
                STATE_PERTURBATION_DATA_DIR
                / f"cpg_{self.gaitnames[self.before_ftype]}_perturbation_{self.tspan[1]}s.csv"
            )

        if not self._if_backward:
            if self.before_ftype == self.after_ftype:
                return (
                    GAIT_DATA_DIR
                    / f"cpg_{self.gaitnames[self.before_ftype]}_{self.tspan[1]}s.csv"
                )
            else:
                formatted_toc = "{:.3f}".format(self.toc)
                formatted_real_toc = "{:.3f}".format(
                    (math.ceil(self.real_toc * 500) * 0.002)
                )
                return (
                    TRANSITION_DATA_DIR
                    / f"cpg_{self.gaitnames[self.before_ftype]}_to_{self.gaitnames[self.after_ftype]}_{formatted_toc}_{formatted_real_toc}.csv"
                )
        else:
            return (
                BACKWARD_DATA_DIR
                / f"cpg_{self.gaitnames[self.before_ftype]}_backward_{self.tspan[1]}s.csv"
            )

    def filter_data(self, data: np.ndarray):
        """filter data to specific format"""
        physical_sim_freq = 500
        gap_num = int(self.time_freq / physical_sim_freq)
        shape = data.shape

        # 如果是用runge_kutta计算所有x,y,z 则返回x即可
        if self._if_mlr_perturbation or self._if_state_perturbation:
            gap_num = 1
        else:
            gap_num = 20
        return (
            data[::gap_num, 0 : self.neuron_nums]
            if shape[1] == self.neuron_nums * 3
            else data[::gap_num, :]
        )

    def export_data(self) -> tuple[np.ndarray, float]:
        """Export data to specified location"""

        t = np.arange(self.tspan[0], self.tspan[1] + self.time_step, self.time_step)
        if self._if_mlr_perturbation or self._if_state_perturbation:
            data = self.runge_kutta_perturbation(self.pos0, t)
        else:
            data, _ = self.runge_kutta_x_and_xdot(self.pos0, t)
        # remove the last row of data
        data = data[:-1, :]

        if self.strategy > 2:
            real_toc = self.wait_toc
        else:
            real_toc = self.toc
        return data, real_toc

    def export_csv(self) -> None:
        """Export data to specified location"""
        data, self.real_toc = self.export_data()

        filtered_data = self.filter_data(data)

        save_path = self.get_output_path()

        self.save_data(file_path=save_path, data=filtered_data)

    def plot_five_gaits(self) -> None:
        """Plot five gaits"""
        import matplotlib.pyplot as plt

        # 获取默认颜色循环
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # self.time_step = 0.01
        # self.time_freq = 100
        t = np.arange(0, self.tspan[1] + self.time_step, self.time_step)
        subfig_num = 6 if self._if_mlr_perturbation else 5
        fig, ax = plt.subplots(subfig_num, 1, figsize=(10, 20), sharex=True)

        for i in range(1, 6):
            self.before_ftype = i
            self.after_ftype = i

            # Initialize parameters
            self.set_neuron_params(ftype=self.before_ftype)
            self.set_link_params(ftype=self.before_ftype)
            self.set_pos0()
            self.set_driving_signal()
            self.get_init_neuron_params(self.before_ftype)

            if self._if_mlr_perturbation:
                self.neuron_params_list = []

                data, _ = self.export_data()
                neuron_params_data = np.array(self.neuron_params_list)

                all_perturbation_data = np.column_stack((t, data, neuron_params_data))
                file_path = self.get_output_path()

                self.save_data(
                    file_path=file_path,
                    data=all_perturbation_data,
                    headers=CPG_MLR_PERTURBATION_HEADERS,
                )

                if i == 1:
                    self.a_list = neuron_params_data[:, 0]
                    ax[0].plot(t, self.a_list, color=default_colors[0])
                    ax[0].set_ylim(9.95, 10.05)
                    ax[0].set_title("a hip of walk")

                for j in range(4):
                    ax[i].plot(t, data[:, j], color=default_colors[j])
                    ax[i].set_title(f"{self.gaitnames[i]}")
                    # set xlim
                    ax[i].set_xlim(10, 15)

                plt.figure()
                plt.plot(data[50000:, 1], data[50000:, 5])

            if self._if_state_perturbation:
                data, _ = self.export_data()

                all_perturbation_data = np.column_stack((t, data))
                file_path = self.get_output_path()

                self.save_data(
                    file_path=file_path,
                    data=all_perturbation_data,
                    headers=CPG_STATE_PERTURBATION_HEADERS,
                )

                for j in range(4):
                    ax[i - 1].plot(t, data[:, j], color=default_colors[j])
                    ax[i - 1].set_title(f"{self.gaitnames[i]}")
                    # set xlim
                    ax[i - 1].set_xlim(10, 15)

        plt.show()


if __name__ == "__main__":
    import pandas

    signal = CPG8Neuron(
        before_ftype=1,
        after_ftype=1,
        total_time=20,
        toc=0.5,
        _if_backward=False,
        _if_mlr_perturbation=True,  # Re-enable MLR perturbation
        # _if_state_perturbation=True,  # Re-enable state perturbation
    )
    signal.plot_five_gaits()
