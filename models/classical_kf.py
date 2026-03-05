# classical_kf.py
from typing import Any, Dict, Optional, Tuple

import torch

from .base_kalman_net import BaseKalmanNet


class ClassicalKalmanFilter(BaseKalmanNet):
    """
    Classical (linear) Kalman Filter that follows BaseKalmanNet's API.

    State evolution:
        x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)
    Observation:
        y_t = H x_t + v_t,       v_t ~ N(0, R)

    Defaults:
        F = I_m
        H = [I_n | 0]  (picks first obs_dim components from the state)
        Q = q_var * I_m
        R = r_var * I_n

    Notes:
    - Uses Joseph's form for covariance update to reduce numerical mess.
    - Respects mask: when mask[t] == 0, the update is skipped for that batch item.

    This class adheres to BaseKalmanNet: forward(observations[, mask, initial_state, return_uncertainty])
    and processes the sequence via repeated calls to _forward_step.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        control_dim: int = 0,
        F: Optional[torch.Tensor] = None,
        H: Optional[torch.Tensor] = None,
        Q: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        q_var: float = 1e-3,
        r_var: float = 1e-2,
        use_joseph: bool = True,
    ):
        super().__init__(state_dim, obs_dim, control_dim)

        # Linear system matrices as buffers so they move with .to(device) and are not optimized
        if F is None:
            F = torch.eye(state_dim)
        if H is None:
            H = torch.zeros(obs_dim, state_dim)
            H[:, :obs_dim] = torch.eye(obs_dim)

        if Q is None:
            Q = torch.eye(state_dim) * q_var
        if R is None:
            R = torch.eye(obs_dim) * r_var

        self.F = torch.nn.Parameter(F.clone())
        self.H = torch.nn.Parameter(H.clone())
        self.Q = torch.nn.Parameter(Q.clone())
        self.R = torch.nn.Parameter(R.clone())
        if control_dim > 0:
            B = torch.zeros(state_dim, control_dim)
            self.B = torch.nn.Parameter(B.clone())

        self.use_joseph = use_joseph

    def _predict(
        self, x: torch.Tensor, P: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        F = self.F.expand(x.shape[0], -1, -1)  # [B, m, m]
        Q = self.Q.expand(x.shape[0], -1, -1)  # [B, m, m]

        x_pred = torch.bmm(F, x.unsqueeze(-1)).squeeze(-1)
        FP = torch.bmm(F, P)
        P_pred = torch.bmm(FP, F.transpose(1, 2)) + Q
        return x_pred, P_pred

    def _update(
        self, x_pred: torch.Tensor, P_pred: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x_pred.shape[0]
        H = self.H.expand(B, -1, -1)  # [B, n, m]
        R = self.R.expand(B, -1, -1)  # [B, n, n]

        # y_pred = H x
        y_pred = torch.bmm(H, x_pred.unsqueeze(-1)).squeeze(-1)
        innov = y - y_pred  # [B, n]

        # S = H P H^T + R
        HP = torch.bmm(H, P_pred)
        S = torch.bmm(HP, H.transpose(1, 2)) + R

        # K = P H^T S^{-1}
        try:
            L = torch.linalg.cholesky(S)
            S_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            S_inv = torch.pinverse(S)

        PHt = torch.bmm(P_pred, H.transpose(1, 2))
        K = torch.bmm(PHt, S_inv)  # [B, m, n]

        # x_new = x_pred + K * innov
        x_new = x_pred + torch.bmm(K, innov.unsqueeze(-1)).squeeze(-1)

        # P_new
        I = (
            torch.eye(self.state_dim, device=x_pred.device)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
        if self.use_joseph:
            # Joseph form: (I - KH) P (I - KH)^T + K R K^T
            KH = torch.bmm(K, H)
            A = I - KH
            AP = torch.bmm(A, P_pred)
            P_new = torch.bmm(AP, A.transpose(1, 2)) + torch.bmm(
                torch.bmm(K, R), K.transpose(1, 2)
            )
        else:
            # Simple form: (I - KH) P
            P_new = torch.bmm(I - torch.bmm(K, H), P_pred)

        return x_new, P_new

    def _forward_step(
        self,
        observation: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        control: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One Kalman step given observation y_t. If mask[b]==0, skip update for that item.

        Args:
            observation: torch.Tensor, shape (B, D)
            mask: Optional[torch.Tensor], shape (B,)
            control: Optional[torch.Tensor], shape (B, D)
        """
        B = observation.shape[0]
        observation.device

        # Predict
        x_prev = self._current_state
        P_prev = self._uncertainty
        x_pred, P_pred = self._predict(x_prev, P_prev)

        # Update
        x_upd, P_upd = self._update(x_pred, P_pred, observation)

        # Optional control injection (minimal): add in state space if it matches
        if control is not None:
            if control.dim() == 1:
                control = control.unsqueeze(-1)
            if control.shape[-1] == self.state_dim:
                x_pred = x_pred + control
            elif hasattr(self, "B"):  # expect B: [state_dim, control_dim]
                x_pred = x_pred + torch.matmul(control, self.B.T)

        if mask is not None:
            m = mask.to(x_upd.dtype).view(B, 1)
            m2 = m.view(B, 1, 1)
            x_next = m * x_upd + (1.0 - m) * x_pred
            P_next = m2 * P_upd + (1.0 - m2) * P_pred
        else:
            x_next, P_next = x_upd, P_upd

        self._current_state = x_next
        self._uncertainty = P_next
        self._previous_obs = observation
        self._initialized = True
        return self._current_state.clone()

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "ClassicalKalmanFilter",
            "covariance_update": "Joseph" if self.use_joseph else "Simple",
            "matrices": {"F": True, "H": True, "Q": True, "R": True},
            "api": "BaseKalmanNet-compatible forward/_forward_step",  # :contentReference[oaicite:2]{index=2}
        }
