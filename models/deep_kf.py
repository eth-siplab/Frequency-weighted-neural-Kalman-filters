# deep_kf.py
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base_kalman_net import BaseKalmanNet
from .kalman_net import MultiLayerPerceptron


class DeepKalmanFilter(BaseKalmanNet):
    """
    Amortized Deep Kalman Filter (KalmanNet-inspired).

    Core idea
    ----------
    Instead of learning K_t with an ad-hoc MLP, we amortize a *local linear model*
    and *noise covariances* from temporal context, then compute the Kalman gain
    analytically:

        x_t = A_t x_{t-1} + b_t + w_t,          w_t ~ N(0, Q_t)
        y_t = H x_t           + v_t,            v_t ~ N(0, R_t)

    The time-varying parameters (A_t, b_t, Q_t, R_t) come from a recurrent encoder
    (a tiny GRU) that ingests [x_{t-1}, y_t]. This is a lightweight, stable version
    of the "learn the Riccati solution" trick used by KalmanNet: we learn covariances
    and residual dynamics, but keep the KF equations to maintain PSD/stability.

    Niceties
    ----------
    - Joseph-form covariance update for numerical stability
    - Cholesky-based inverses for S = H P H^T + R
    - Optional low-rank corrections for Q_t and R_t
    - Residual dynamics A_t = I + ΔA_t with either diagonal gate or low-rank UV^T
    - Proper mask handling (skip update when mask[b]==0)
    - Optionally detach gain path (S^{-1}) from gradients for extra stability

    API contract
    -------------
    Matches BaseKalmanNet:
      forward(observations[, mask, initial_state, return_uncertainty]) -> [B, T, state_dim]
      _forward_step(observation, mask) -> [B, state_dim]
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        control_dim: int = 0,
        hidden_dim: int = 256,
        enc_layers: int = 2,
        # Residual dynamics options
        rank_dyn: int = 4,  # if >0, A_t = I + U V^T; else diagonal gate
        dyn_scale: float = 0.1,  # scale for UV^T or diag gate
        # Low-rank noise structure
        rank_q: int = 2,
        rank_r: int = 1,
        # Variance floor for numerical sanity
        min_var: float = 1e-5,
        # Training stability
        detach_gain_grad: bool = True,
        # IIR innovation filter orders (paper Alg. 2)
        iir_p: int = 3,  # feedforward (FIR) order
        iir_q: int = 3,  # feedback (pole) order
        # Nonlinear observation model (for datasets like EuRoC with IMU)
        use_nonlinear_obs: bool = False,
    ):
        super().__init__(state_dim, obs_dim, control_dim)
        self.hidden_dim = hidden_dim
        self.rank_dyn = int(rank_dyn)
        self.rank_q = int(rank_q)
        self.rank_r = int(rank_r)
        self.dyn_scale = float(dyn_scale)
        self.min_var = float(min_var)
        self.detach_gain_grad = bool(detach_gain_grad)

        # Observation model: either learned MLP h(x) or fixed linear H
        if use_nonlinear_obs:
            h_hidden = max(32, int(math.sqrt(state_dim * obs_dim)) * 4)
            self.h_net = MultiLayerPerceptron(
                state_dim, h_hidden, h_hidden, output_dim=obs_dim
            )
            self.h_net.append(nn.Tanh())
        else:
            self.h_net = None

        # Linear observation matrix H = [I_obs | 0] used for covariance updates
        # When h_net is active this serves as a first-order approximation
        H = torch.zeros(obs_dim, state_dim)
        H[:, :obs_dim] = torch.eye(obs_dim)
        self.register_buffer("H", H)

        # Encoder: amortize parameters from context [x_{t-1}, y_t]
        enc_in = state_dim + obs_dim + control_dim
        self.enc = nn.GRU(enc_in, hidden_dim, num_layers=enc_layers, batch_first=True)

        # Heads from h_t -> parameters
        # Dynamics: either low-rank UV^T or diagonal gate
        if self.rank_dyn > 0:
            self.head_U = nn.Linear(hidden_dim, state_dim * self.rank_dyn)
            self.head_V = nn.Linear(hidden_dim, state_dim * self.rank_dyn)
        else:
            self.head_A_diag = nn.Linear(hidden_dim, state_dim)  # diag gate Δa

        self.head_b = nn.Linear(hidden_dim, state_dim)

        # Process noise Q_t
        self.head_q_diag = nn.Linear(hidden_dim, state_dim)
        if self.rank_q > 0:
            self.head_q_lr = nn.Linear(hidden_dim, state_dim * self.rank_q)

        # Observation noise R_t
        self.head_r_diag = nn.Linear(hidden_dim, obs_dim)
        if self.rank_r > 0:
            self.head_r_lr = nn.Linear(hidden_dim, obs_dim * self.rank_r)

        # --- IIR innovation filter (paper Sec. 2.2, Alg. 2) ---
        # Learnable coefficients for causal IIR: Φ(z) = B(z⁻¹)/A(z⁻¹)
        #   B(z⁻¹) = Σ_{i=0}^{p} b_i z^{-i}   (feedforward / FIR)
        #   A(z⁻¹) = 1 + Σ_{j=1}^{q} a_j z^{-j} (feedback / poles)
        # Shared across measurement channels, applied componentwise.
        self.iir_p = int(iir_p)
        self.iir_q = int(iir_q)

        # Feedforward coefficients b_0..b_p  (init: identity passthrough b_0=1)
        self.iir_b = nn.Parameter(torch.zeros(self.iir_p + 1))
        self.iir_b.data[0] = 1.0

        # Feedback coefficients a_1..a_q stored as unconstrained params.
        # Stability: we reparameterize via tanh to keep pole magnitudes < 1.
        self.iir_a_raw = nn.Parameter(torch.zeros(self.iir_q))

        # IIR internal buffer, set in reset_state
        self._iir_buf: Optional[torch.Tensor] = None  # [B, max(p,q)+1, obs_dim]

        # Recurrent hidden state cached across steps
        self._enc_h: Optional[torch.Tensor] = None  # [1, B, hidden_dim]

    # ----------------------------
    # Utilities
    # ----------------------------
    def _iir_filter(self, innov: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable IIR filter to innovation (paper Alg. 2, Eqs. 16-18).

        Cascaded realization:
            s_k = -Σ_{j=1}^{q} a_j s_{k-j} + Δy_k     (pole recursion)
            Δ̃y_k = Σ_{i=0}^{p} b_i s_{k-i}             (FIR stage)

        Stability: feedback coefficients constrained via tanh so |a_j| < 1.
        The filter is shared across channels, applied componentwise.

        Args:
            innov: [B, obs_dim] raw innovation Δy_t
        Returns:
            [B, obs_dim] filtered innovation Δ̃y_t
        """
        # Stable feedback coefficients: tanh keeps |a_j| < 1
        a = torch.tanh(self.iir_a_raw)  # [q]

        # Pole recursion: s_k = Δy_k - Σ a_j * s_{k-j}
        # _iir_buf[:, 0, :] is the most recent, [:, j, :] is j steps ago
        s_k = innov.clone()
        for j in range(self.iir_q):
            s_k = s_k - a[j] * self._iir_buf[:, j, :]

        # FIR stage: Δ̃y_k = Σ b_i * s_{k-i}
        # s_{k-0} = s_k (just computed), s_{k-i} = _iir_buf[:, i-1, :] for i>=1
        filtered = self.iir_b[0] * s_k
        for i in range(1, self.iir_p + 1):
            filtered = filtered + self.iir_b[i] * self._iir_buf[:, i - 1, :]

        # Shift buffer: push s_k as newest, drop oldest
        self._iir_buf = torch.cat(
            [s_k.unsqueeze(1), self._iir_buf[:, :-1, :]], dim=1
        )

        return filtered

    def _build_A(self, h: torch.Tensor) -> torch.Tensor:
        """
        Build A_t = I + ΔA_t from encoder state h.
        If rank_dyn>0: ΔA_t = dyn_scale * U V^T
        else:          ΔA_t = dyn_scale * diag(tanh(a))
        """
        B = h.shape[0]
        I = torch.eye(self.state_dim, device=h.device).unsqueeze(0).expand(B, -1, -1)

        if self.rank_dyn > 0:
            U = self.head_U(h).view(B, self.state_dim, self.rank_dyn)
            V = self.head_V(h).view(B, self.state_dim, self.rank_dyn)
            # Keep correction small; use tanh to cap magnitude
            U = torch.tanh(U)
            V = torch.tanh(V)
            dA = self.dyn_scale * torch.bmm(U, V.transpose(1, 2))  # [B, m, m]
            A = I + dA
        else:
            a = torch.tanh(self.head_A_diag(h))  # [-1,1]
            A = I + self.dyn_scale * torch.diag_embed(a)

        return A

    def _build_Q(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        q_diag = torch.nn.functional.softplus(self.head_q_diag(h)) + self.min_var
        Q = torch.diag_embed(q_diag)  # [B, m, m]
        if self.rank_q > 0:
            Uq = self.head_q_lr(h).view(B, self.state_dim, self.rank_q)
            Q = Q + torch.bmm(Uq, Uq.transpose(1, 2))
        return Q

    def _build_R(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        r_diag = torch.nn.functional.softplus(self.head_r_diag(h)) + self.min_var
        R = torch.diag_embed(r_diag)  # [B, n, n]
        if self.rank_r > 0:
            Ur = self.head_r_lr(h).view(B, self.obs_dim, self.rank_r)
            R = R + torch.bmm(Ur, Ur.transpose(1, 2))
        return R

    def _predict(
        self, x_prev: torch.Tensor, P_prev: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict step using amortized A_t, b_t, Q_t from encoder state h.
        """
        x_prev.shape[0]
        A = self._build_A(h)  # [B, m, m]
        b = self.head_b(h)  # [B, m]
        Q = self._build_Q(h)  # [B, m, m]

        # x_pred = A x_prev + b
        x_pred = torch.bmm(A, x_prev.unsqueeze(-1)).squeeze(-1) + b

        # P_pred = A P A^T + Q
        AP = torch.bmm(A, P_prev)
        P_pred = torch.bmm(AP, A.transpose(1, 2)) + Q
        return x_pred, P_pred

    def _kf_gain(self, P_pred: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Compute K_t = P_pred H^T (H P_pred H^T + R)^{-1}
        Optionally detach the inverse path from gradients for stability.
        """
        B = P_pred.shape[0]
        H = self.H.expand(B, -1, -1)  # [B, n, m]

        P_used = P_pred.detach() if self.detach_gain_grad else P_pred

        HP = torch.bmm(H, P_used)
        S = torch.bmm(HP, H.transpose(1, 2)) + R  # [B, n, n]

        try:
            L = torch.linalg.cholesky(S)
            S_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            S_inv = torch.pinverse(S)

        PHt = torch.bmm(P_used, H.transpose(1, 2))
        K = torch.bmm(PHt, S_inv)  # [B, m, n]
        return K

    def _cov_update_joseph(
        self, P_pred: torch.Tensor, K: torch.Tensor, R: torch.Tensor
    ) -> torch.Tensor:
        B = P_pred.shape[0]
        H = self.H.expand(B, -1, -1)
        I = (
            torch.eye(self.state_dim, device=P_pred.device)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
        KH = torch.bmm(K, H)
        A = I - KH
        AP = torch.bmm(A, P_pred)
        return torch.bmm(AP, A.transpose(1, 2)) + torch.bmm(
            torch.bmm(K, R), K.transpose(1, 2)
        )

    # ----------------------------
    # BaseKalmanNet overrides
    # ----------------------------
    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        /,
        initial_state: Optional[torch.Tensor] = None,
        initial_covariance: Optional[torch.Tensor] = None,
    ):
        super().reset_state(
            batch_size,
            device,
            initial_state=initial_state,
            initial_covariance=initial_covariance,
        )
        # Zero hidden state for the amortization encoder
        self._enc_h = torch.zeros(
            self.enc.num_layers, batch_size, self.hidden_dim, device=device
        )
        # Zero IIR internal buffer: [B, max(p,q)+1, obs_dim]
        buf_len = max(self.iir_p, self.iir_q) + 1
        self._iir_buf = torch.zeros(
            batch_size, buf_len, self.obs_dim, device=device
        )

    def _forward_step(
        self,
        observation: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        control: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One amortized DKF step:
          1) Encode context z_t = GRU([x_{t-1}, y_t], h_{t-1})
          2) Predict with A_t, b_t, Q_t
          3) Compute K_t from P_pred and R_t
          4) Joseph-form covariance update
          5) Apply mask if present

        Args:
            observation: [B, n]
            mask: [B, 1]
            control: [B, m]
        Returns:
            [B, d]
        """
        B = observation.shape[0]
        observation.device

        # 1) Encode context
        if control is not None:
            enc_in = torch.cat([self._current_state, observation, control], dim=-1)
        else:
            enc_in = torch.cat([self._current_state, observation], dim=-1)
        enc_in = enc_in.unsqueeze(1)  # [B,1,m+n]
        enc_out, h_new = self.enc(
            enc_in, self._enc_h
        )  # enc_out: [B,1,H]; h_new: [1,B,H]
        h_t = enc_out.squeeze(1)  # [B,H]
        self._enc_h = h_new

        # 2) Predict
        x_prev = self._current_state
        P_prev = self._uncertainty
        x_pred, P_pred = self._predict(x_prev, P_prev, h_t)

        # 3) Gain from learned R_t
        R_t = self._build_R(h_t)
        K_t = self._kf_gain(P_pred, R_t)

        # 4) Update state and covariance
        # Always use the linear H for the innovation (paper Alg. 3, line 9).
        # h_net is only used in the spectral loss (Alg. 1).
        H = self.H.expand(B, -1, -1)
        y_pred = torch.bmm(H, x_pred.unsqueeze(-1)).squeeze(-1)
        innov = observation - y_pred

        # 4a) IIR innovation filtering (paper Alg. 2)
        innov_filtered = self._iir_filter(innov)

        x_upd = x_pred + torch.bmm(K_t, innov_filtered.unsqueeze(-1)).squeeze(-1)
        P_upd = self._cov_update_joseph(P_pred, K_t, R_t)

        # 5) Masked update
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
            "model_type": "DeepKalmanFilter",
            "param_amortization": {
                "encoder": "GRU(x_{t-1}, y_t) -> h_t",
                "A_t": "I + ΔA_t (diag or low-rank)",
                "b_t": "learned bias per step",
                "Q_t": f"diag(+ optional low-rank rank_q={self.rank_q})",
                "R_t": f"diag(+ optional low-rank rank_r={self.rank_r})",
            },
            "gain": "computed analytically from P_pred, R_t (Cholesky S^{-1})",
            "covariance_update": "Joseph form",
            "iir_innovation_filter": {
                "feedforward_order_p": self.iir_p,
                "feedback_order_q": self.iir_q,
                "stability": "tanh reparameterization on feedback coefficients",
            },
            "stability": {
                "min_var": self.min_var,
                "detach_gain_grad": self.detach_gain_grad,
                "dyn_scale": self.dyn_scale,
            },
            "api": "BaseKalmanNet-compatible forward/_forward_step",
        }
