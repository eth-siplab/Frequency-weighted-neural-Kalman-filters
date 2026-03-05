import warnings
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.integrate import odeint
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def standard_normal(
    rng: Optional[np.random.Generator] = None,
    loc: float = 0.0,
    scale: float = 1.0,
    size: Tuple[int, ...] = (1,),
):
    if rng is None:
        return np.random.normal(loc, scale, size=size)
    else:
        return loc + scale * rng.normal(size=size)


class LorenzDataset(Dataset):
    """
    Lorenz Attractor Dataset for KalmanNet training.

    Lorenz system:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

    Standard parameters: σ=10, ρ=28, β=8/3
    """

    def __init__(
        self,
        num_trajectories: int = 1000,
        sequence_length: int = 100,
        dt: float = 0.01,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        process_noise_std: float = 0.1,
        measurement_noise_std: float = 0.5,
        partial_obs: bool = True,
        device: str = "cpu",
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize Lorenz dataset.

        Args:
            num_trajectories: Number of trajectories to generate
            sequence_length: Length of each trajectory
            dt: Time step for integration
            sigma, rho, beta: Lorenz system parameters
            process_noise_std: Standard deviation of process noise
            measurement_noise_std: Standard deviation of measurement noise
            partial_obs: If True, observe only [x, y]. If False, observe [x, y, z]
            device: torch device
        """
        self.num_trajectories = num_trajectories
        self.sequence_length = sequence_length
        self.dt = dt
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.partial_obs = partial_obs
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        # State dimension is always 3 [x, y, z]
        self.state_dim = 3
        # Observation dimension depends on partial_obs
        self.obs_dim = 2 if partial_obs else 3

        # Generate all trajectories
        self.states, self.observations = self._generate_trajectories()

    def lorenz_dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        """Lorenz system dynamics"""
        x, y, z = state
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return np.array([dxdt, dydt, dzdt])

    def _generate_single_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single Lorenz trajectory"""
        # Random initial condition
        initial_state = standard_normal(self.rng, size=(3,), scale=5.0)

        # Time points
        t = np.linspace(0, self.sequence_length * self.dt, self.sequence_length)

        # Integrate Lorenz system
        true_trajectory = odeint(self.lorenz_dynamics, initial_state, t)

        # Add process noise
        process_noise = standard_normal(
            rng=self.rng,
            loc=0.0,
            scale=self.process_noise_std,
            size=true_trajectory.shape,
        )
        noisy_trajectory = true_trajectory + process_noise

        # Generate observations
        if self.partial_obs:
            # Observe only x and y coordinates
            observations = noisy_trajectory[:, :2].copy()
        else:
            # Observe all coordinates
            observations = noisy_trajectory.copy()

        # Add measurement noise
        measurement_noise = standard_normal(
            rng=self.rng,
            loc=0.0,
            scale=self.measurement_noise_std,
            size=observations.shape,
        )
        observations += measurement_noise

        return true_trajectory, observations

    def _generate_trajectories(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all trajectories"""
        all_states = []
        all_observations = []

        for i in range(self.num_trajectories):
            states, obs = self._generate_single_trajectory()
            all_states.append(states)
            all_observations.append(obs)

        # Convert to tensors
        states_tensor = torch.tensor(
            np.array(all_states), dtype=torch.float32, device=self.device
        )
        obs_tensor = torch.tensor(
            np.array(all_observations), dtype=torch.float32, device=self.device
        )

        return states_tensor, obs_tensor

    def __len__(self) -> int:
        return self.num_trajectories

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "states": self.states[idx],  # [seq_len, 3]
            "observations": self.observations[idx],  # [seq_len, obs_dim]
        }

    def get_state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get linearized state space matrices around equilibrium"""
        # Linearization around one of the equilibria
        if self.partial_obs:
            H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Observe x, y
        else:
            H = np.eye(3)  # Observe all states

        # Approximate F matrix (highly simplified for Lorenz)
        F = np.array(
            [
                [1 - self.sigma * self.dt, self.sigma * self.dt, 0],
                [self.rho * self.dt, 1 - self.dt, 0],
                [0, 0, 1 - self.beta * self.dt],
            ]
        )

        return F, H


class PendulumDataset(Dataset):
    """
    Nonlinear Pendulum Dataset for KalmanNet training.

    Pendulum dynamics:
    dθ/dt = ω
    dω/dt = -(g/L)sin(θ) - (b/m)ω + u/m

    State: [θ, ω] (angle, angular velocity)
    Observations: [x, y] = [L*sin(θ), -L*cos(θ)] (Cartesian position)
    """

    def __init__(
        self,
        num_trajectories: int = 1000,
        sequence_length: int = 100,
        dt: float = 0.02,
        g: float = 9.81,
        L: float = 1.0,
        b: float = 0.1,
        m: float = 1.0,
        process_noise_std: float = 0.05,
        measurement_noise_std: float = 0.1,
        control_amplitude: float = 0.5,
        device: str = "cpu",
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize pendulum dataset.

        Args:
            num_trajectories: Number of trajectories to generate
            sequence_length: Length of each trajectory
            dt: Time step for integration
            g: Gravitational acceleration
            L: Pendulum length
            b: Damping coefficient
            m: Mass
            process_noise_std: Standard deviation of process noise
            measurement_noise_std: Standard deviation of measurement noise
            control_amplitude: Amplitude of random control input
            device: torch device
        """
        self.num_trajectories = num_trajectories
        self.sequence_length = sequence_length
        self.dt = dt
        self.g = g
        self.L = L
        self.b = b
        self.m = m
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.control_amplitude = control_amplitude
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        # State: [θ, ω], Obs: [x, y] = [L*sin(θ), -L*cos(θ)]
        self.state_dim = 2
        self.obs_dim = 2

        # Generate all trajectories
        self.states, self.observations, self.controls = self._generate_trajectories()

    def pendulum_dynamics(
        self, state: np.ndarray, t: float, control: float
    ) -> np.ndarray:
        """Pendulum dynamics with damping and control"""
        theta, omega = state
        dthetadt = omega
        domegadt = (
            -(self.g / self.L) * np.sin(theta)
            - (self.b / self.m) * omega
            + control / self.m
        )
        return np.array([dthetadt, domegadt])

    def _generate_single_trajectory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a single pendulum trajectory"""
        # Random initial condition
        initial_theta = self.rng.uniform(-np.pi, np.pi)  # Random angle
        initial_omega = self.rng.uniform(-2.0, 2.0)  # Random angular velocity
        initial_state = np.array([initial_theta, initial_omega])

        # Generate random control sequence
        controls = standard_normal(
            rng=self.rng,
            loc=0,
            scale=self.control_amplitude,
            size=(self.sequence_length,),
        )

        # Integrate using Euler method (more control over noise injection)
        states = np.zeros((self.sequence_length, 2))
        states[0] = initial_state

        for i in range(1, self.sequence_length):
            # Dynamics
            dstate = self.pendulum_dynamics(states[i - 1], i * self.dt, controls[i - 1])

            # Euler step
            states[i] = states[i - 1] + self.dt * dstate

            # Add process noise
            process_noise = standard_normal(
                rng=self.rng, loc=0, scale=self.process_noise_std, size=(2,)
            )
            states[i] += process_noise

            # Keep angle in reasonable range
            states[i, 0] = np.mod(states[i, 0] + np.pi, 2 * np.pi) - np.pi

        # Generate observations: Cartesian coordinates
        observations = np.zeros((self.sequence_length, 2))
        observations[:, 0] = self.L * np.sin(states[:, 0])  # x = L*sin(θ)
        observations[:, 1] = -self.L * np.cos(states[:, 0])  # y = -L*cos(θ)

        # Add measurement noise
        measurement_noise = standard_normal(
            rng=self.rng,
            loc=0,
            scale=self.measurement_noise_std,
            size=observations.shape,
        )
        observations += measurement_noise

        return states, observations, controls

    def _generate_trajectories(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all trajectories"""
        all_states = []
        all_observations = []
        all_controls = []

        for i in range(self.num_trajectories):
            states, obs, controls = self._generate_single_trajectory()
            all_states.append(states)
            all_observations.append(obs)
            all_controls.append(controls)

        # Convert to tensors
        states_tensor = torch.tensor(
            np.array(all_states), dtype=torch.float32, device=self.device
        )
        obs_tensor = torch.tensor(
            np.array(all_observations), dtype=torch.float32, device=self.device
        )
        controls_tensor = torch.tensor(
            np.array(all_controls), dtype=torch.float32, device=self.device
        )

        return states_tensor, obs_tensor, controls_tensor

    def __len__(self) -> int:
        return self.num_trajectories

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "states": self.states[idx],  # [seq_len, 2]
            "observations": self.observations[idx],  # [seq_len, 2]
            "controls": self.controls[idx].unsqueeze(1),  # [seq_len, 1]
        }

    def get_state_space_matrices(
        self, theta: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get linearized state space matrices around given angle"""
        # Linearized F matrix around θ
        F = np.array(
            [
                [1.0, self.dt],
                [
                    -(self.g / self.L) * np.cos(theta) * self.dt,
                    1 - self.b * self.dt / self.m,
                ],
            ]
        )

        # Observation Jacobian H = d/dstate [L*sin(θ), -L*cos(θ)]
        H = np.array([[self.L * np.cos(theta), 0.0], [self.L * np.sin(theta), 0.0]])

        return F, H
