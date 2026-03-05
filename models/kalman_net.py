# kalman_net_v2.py

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_kalman_net import BaseKalmanNet


class MultiLayerPerceptron(nn.Sequential):
    def __init__(self, input_dim: int, *hidden_dims: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.append(nn.Linear(input_dim, hidden_dims[0]))
        self.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            self.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.append(nn.ReLU())
        self.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(
        self, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return super().forward(torch.cat([x, u], dim=1) if u is not None else x)


# ---------- synthetic nonlinear model used in the paper (eq. 17) ----------
# f(x) = α sin(β x + φ) + δ   (applied elementwise)
class SinusoidalTransition(nn.Module):
    def __init__(
        self,
        alpha: float = 0.9,
        beta: float = 1.1,
        phi: float = 0.1 * math.pi,
        delta: float = 0.01,
    ):
        super().__init__()
        self.alpha, self.beta, self.phi, self.delta = alpha, beta, phi, delta

    def forward(
        self, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return (
            self.alpha * torch.sin(self.beta * x + self.phi) + self.delta
        )  # elementwise


# h(x) = a (b x + c)^2        (applied elementwise)
class QuadraticEmission(nn.Module):
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 0.0):
        super().__init__()
        self.a, self.b, self.c = a, b, c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * (self.b * x + self.c) ** 2


# ---------- Lorenz transition used in the paper (eqs. 20–21) ----------
# Discrete-time approximation: F(x) ≈ I + Σ_{j=1..J} (A(x) Δt)^j / j!, then x_{t+1} = F(x_t) x_t
class LorenzTransition(nn.Module):
    def __init__(self, dt: float = 0.02, order: int = 5):
        super().__init__()
        self.dt, self.order = float(dt), int(order)

    def _A(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3]; returns A(x): [B, 3, 3]
        B = x.shape[0]
        A = x.new_zeros(B, 3, 3)
        x1 = x[:, 0]
        A[:, 0, 0] = -10.0
        A[:, 0, 1] = 10.0
        A[:, 1, 0] = 28.0
        A[:, 1, 1] = -1.0
        A[:, 1, 2] = -x1
        A[:, 2, 1] = x1
        A[:, 2, 2] = -8.0 / 3.0
        return A

    def forward(
        self, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x.shape[-1] == 3, "LorenzTransition expects state_dim=3"
        B = x.shape[0]
        I = torch.eye(3, device=x.device, dtype=x.dtype).expand(B, 3, 3)
        A_dt = self._A(x) * self.dt
        F_ = I.clone()
        term = I.clone()
        for j in range(1, self.order + 1):
            term = term @ A_dt / j  # term = (A*dt)^j / j!
            F_ = F_ + term
        return torch.bmm(F_, x.unsqueeze(-1)).squeeze(-1)


# ---------- Spherical emission used in the paper’s Lorenz experiment ----------
class SphericalEmission(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3] -> y: [B,3] with (r, theta, phi)
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        eps = 1e-8
        r = torch.sqrt(x1**2 + x2**2 + x3**2 + eps)
        theta = torch.atan2(x2, x1)
        phi = torch.atan2(x3, torch.sqrt(x1**2 + x2**2 + eps))
        return torch.stack([r, theta, phi], dim=1)


class KalmanNet(BaseKalmanNet):
    """
    KalmanNet Architecture #2 - Separate GRUs with FC_1 to FC_7 layers.

    Paper: "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics"
    ArXiv: https://arxiv.org/abs/2107.10043
    GitHub: https://github.com/KalmanNet/KalmanNet_TSP

    This is Architecture #2 from Fig. 4 with the full FC_1 to FC_7 implementation
    from KalmanNet_nn.py and GSSFiltering/dnn.py.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        control_dim: int = 0,
        in_mult: int = 5,
        out_mult: int = 40,
        transition_model: str = "mlp",  # options: "mlp", "sinusoid", "lorenz", "identity"
        emission_model: str = "mlp",  # options: "mlp", "quadratic", "spherical", "identity"
    ):
        super().__init__(state_dim, obs_dim, control_dim)

        self.m = state_dim
        self.n = obs_dim
        self.k = control_dim
        self.in_mult = in_mult
        self.out_mult = out_mult

        # ---------------------------
        # Build transition f(x[,u])
        # ---------------------------
        t = transition_model.lower()
        if t == "mlp":
            # f takes [x,u] if control_dim>0 else just x
            f_in = self.m + (self.k if self.k and self.k > 0 else 0)
            # two modest hidden layers; scale with dims but keep it sane
            f_h1 = max(32, self.in_mult * self.m)
            f_h2 = max(32, self.in_mult * self.m)
            self.f = MultiLayerPerceptron(f_in, f_h1, f_h2, output_dim=self.m)
            self.f_uses_control = self.k is not None and self.k > 0
        elif t == "sinusoid":
            self.f = SinusoidalTransition()
            self.f_uses_control = False
        elif t == "lorenz":
            # Discrete-time Taylor approximation of the Lorenz flow (expects m==3)
            self.f = LorenzTransition(dt=0.02, order=5)
            self.f_uses_control = False
        elif t == "identity":
            self.f = nn.Identity()
            self.f_uses_control = False
        else:
            raise ValueError(f"Unknown transition_model '{transition_model}'")

        # ---------------------------
        # Build emission h(x)
        # ---------------------------
        e = emission_model.lower()
        if e == "mlp":
            # h maps state -> observation
            h_h1 = max(32, self.in_mult * self.n)
            h_h2 = max(32, self.in_mult * self.n)
            self.h = MultiLayerPerceptron(self.m, h_h1, h_h2, output_dim=self.n)
        elif e == "quadratic":
            self.h = QuadraticEmission()
        elif e == "spherical":
            self.h = SphericalEmission()
        elif e == "identity":
            self.h = nn.Identity()
        else:
            raise ValueError(f"Unknown emission_model '{emission_model}'")

        # GRU dimensions from KalmanNet_nn.py
        self.d_hidden_Q = self.m**2
        self.d_hidden_Sigma = self.m**2
        self.d_hidden_S = self.n**2

        # Three GRUs as in Architecture #2
        self.GRU_Q = nn.GRU(self.m * in_mult, self.d_hidden_Q, batch_first=False)
        self.GRU_Sigma = nn.GRU(
            self.d_hidden_Q + self.m * in_mult, self.d_hidden_Sigma, batch_first=False
        )
        self.GRU_S = nn.GRU(
            self.n**2 + 2 * self.n * in_mult, self.d_hidden_S, batch_first=False
        )

        # FC layers exactly as in official implementation

        # FC1: From GSSFiltering/dnn.py lines 62-65
        self.FC1 = nn.Sequential(nn.Linear(self.d_hidden_Sigma, self.n**2), nn.ReLU())

        # FC2: From GSSFiltering/dnn.py lines 67-73 - THIS IS THE KALMAN GAIN OUTPUT
        d_hidden_FC2 = (self.d_hidden_S + self.d_hidden_Sigma) * out_mult
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_hidden_S + self.d_hidden_Sigma, d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(d_hidden_FC2, self.n * self.m),
        )

        # FC3: From GSSFiltering/dnn.py lines 75-79
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_hidden_S + self.n * self.m, self.m**2), nn.ReLU()
        )

        # FC4: From GSSFiltering/dnn.py lines 81-85
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma + self.m**2, self.d_hidden_Sigma), nn.ReLU()
        )

        # FC5: From GSSFiltering/dnn.py lines 87-91
        self.FC5 = nn.Sequential(nn.Linear(self.m, self.m * in_mult), nn.ReLU())

        # FC6: From GSSFiltering/dnn.py lines 93-97
        self.FC6 = nn.Sequential(nn.Linear(self.m, self.m * in_mult), nn.ReLU())

        # FC7: From GSSFiltering/dnn.py lines 99-103
        self.FC7 = nn.Sequential(nn.Linear(2 * self.n, 2 * self.n * in_mult), nn.ReLU())

        # Hidden states for GRUs
        self.h_Q = None
        self.h_Sigma = None
        self.h_S = None

        # Previous states for feature computation
        self._state_prev = None
        self._obs_pred_prev = None

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        initial_state: Optional[torch.Tensor] = None,
        initial_covariance: Optional[torch.Tensor] = None,
    ):
        """Reset all internal states."""
        super().reset_state(
            batch_size,
            device,
            initial_state=initial_state,
            initial_covariance=initial_covariance,
        )
        self.h_Q = torch.zeros(1, batch_size, self.d_hidden_Q, device=device)
        self.h_Sigma = torch.zeros(1, batch_size, self.d_hidden_Sigma, device=device)
        self.h_S = torch.zeros(1, batch_size, self.d_hidden_S, device=device)
        self._state_prev = torch.zeros(batch_size, self.state_dim, device=device)
        self._obs_pred_prev = torch.zeros(batch_size, self.obs_dim, device=device)

    def _compute_all_features(
        self, obs: torch.Tensor, control: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute all four features F1-F4 as per KalmanNet_nn.py step_KGain_est method.

        From KalmanNet_nn.py lines 90-105:
        - obs_diff = y - y_previous (F1)
        - obs_innov_diff = y - y_hat (F2)
        - fw_evol_diff = x_hat - x_hat_previous (F3)
        - fw_update_diff = x_hat - x_pred_previous (F4)
        """
        obs.shape[0]
        obs.device

        # Current predictions
        obs_pred = self.h(self._current_state)

        # F1: Observation difference - y_t - y_{t-1}
        obs_diff = obs - self._previous_obs

        # F2: Innovation difference - y_t - ŷ_{t|t-1}
        obs_innov_diff = obs - obs_pred

        # F3: Forward evolution difference - x̂_{t|t} - x̂_{t-1|t-1}
        fw_evol_diff = self._current_state - self._state_prev

        # F4: Forward update difference - x̂_{t|t} - x̂_{t|t-1}
        # For this we need the predicted state
        state_pred = (
            self.f(self._state_prev, control if self.f_uses_control else None)
            if self._initialized
            else self._current_state
        )
        fw_update_diff = self._current_state - state_pred

        # Normalize as in original implementation - from KalmanNet_nn.py lines 96-99
        obs_diff = F.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        obs_innov_diff = F.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12)
        fw_evol_diff = F.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        fw_update_diff = F.normalize(fw_update_diff, p=2, dim=1, eps=1e-12)

        return obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff

    def _kalman_gain_estimation(
        self,
        obs_diff: torch.Tensor,
        obs_innov_diff: torch.Tensor,
        fw_evol_diff: torch.Tensor,
        fw_update_diff: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate Kalman Gain using the complete FC1-FC7 + GRU architecture.

        This implements the exact flow from GSSFiltering/dnn.py forward method lines 116-168.
        """

        def expand_dim(x):
            # From GSSFiltering/dnn.py lines 118-121
            expanded = torch.empty(1, x.shape[0], x.shape[-1], device=x.device)
            expanded[0, :, :] = x
            return expanded

        # Expand dimensions as per original code
        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        ####################
        ### Forward Flow ###
        ####################

        # FC5 → Q-GRU - From GSSFiltering/dnn.py lines 127-130
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        # FC6 → Sigma-GRU - From GSSFiltering/dnn.py lines 137-140
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC1 - From GSSFiltering/dnn.py lines 142-143
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC7 - From GSSFiltering/dnn.py lines 145-146
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        # S-GRU - From GSSFiltering/dnn.py lines 149-150
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        # FC2 - Kalman Gain computation - From GSSFiltering/dnn.py lines 153-154
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC3 - From GSSFiltering/dnn.py lines 159-160
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC4 - From GSSFiltering/dnn.py lines 162-163
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # Update Sigma hidden state - From GSSFiltering/dnn.py line 166
        self.h_Sigma = out_FC4

        # Return Kalman Gain reshaped - From GSSFiltering/dnn.py line 168
        return out_FC2.reshape(out_FC2.shape[1], self.m, self.n)

    def _forward_step(
        self,
        observation: torch.Tensor,
        mask: torch.Tensor = None,
        control: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Process a single observation step with full KalmanNet V2 architecture.

        Args:
            observation: [batch, obs_dim]
            mask: [batch] - validity mask
            control: [batch, control_dim] - control input

        Returns:
            filtered_state: [batch, state_dim]
        """
        batch_size = observation.shape[0]
        device = observation.device

        if mask is None:
            mask = torch.ones(batch_size, device=device)

        # Predict step
        if self._initialized:
            # Store previous state for feature computation
            self._state_prev = self._current_state.clone()
            self._current_state = self.f(
                self._current_state, control if self.f_uses_control else None
            )
        else:
            # Initialize
            if self.state_dim >= self.obs_dim:
                self._current_state[:, : self.obs_dim] = observation
            self._state_prev = self._current_state.clone()
            self._initialized = True

        # Compute all features F1-F4
        obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff = (
            self._compute_all_features(observation, control)
        )

        # Estimate Kalman Gain using full architecture
        kalman_gain = self._kalman_gain_estimation(
            obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff
        )

        # Update step
        obs_pred = self.h(self._current_state)
        innovation = observation - obs_pred

        # Apply mask
        masked_innovation = innovation * mask.unsqueeze(-1)

        # Kalman update
        update = torch.bmm(kalman_gain, masked_innovation.unsqueeze(-1)).squeeze(-1)
        self._current_state = self._current_state + update

        # Update uncertainty (simplified)
        if self._uncertainty is not None:
            H = (
                torch.eye(self.obs_dim, self.state_dim, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            KH = torch.bmm(kalman_gain, H)
            I = (
                torch.eye(self.state_dim, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            self._uncertainty = torch.bmm(I - KH, self._uncertainty)

        # Store for next iteration
        self._previous_obs = observation.clone()

        return self._current_state.clone()

    def get_model_info(self) -> Dict[str, Any]:
        """Return model-specific information."""
        return {
            "model_type": "KalmanNet_V2",
            "architecture": "Separate GRUs with FC_1 to FC_7 layers",
            "gru_dimensions": {
                "Q_GRU": self.d_hidden_Q,
                "Sigma_GRU": self.d_hidden_Sigma,
                "S_GRU": self.d_hidden_S,
            },
            "multipliers": {"in_mult": self.in_mult, "out_mult": self.out_mult},
            "features": [
                "F1_obs_diff",
                "F2_innovation_diff",
                "F3_evol_diff",
                "F4_update_diff",
            ],
            "fc_layers": ["FC1", "FC2_KalmanGain", "FC3", "FC4", "FC5", "FC6", "FC7"],
            "paper": "KalmanNet: Neural Network Aided Kalman Filtering (Architecture #2)",
            "arxiv": "https://arxiv.org/abs/2107.10043",
            "github": "https://github.com/KalmanNet/KalmanNet_TSP",
        }
