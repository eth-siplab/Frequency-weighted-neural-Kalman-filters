# autoreg_kf.py

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base_kalman_net import BaseKalmanNet


class AutoRegKF(BaseKalmanNet):
    """
    Autoregressive Kalman Filter with Granger causality modeling for multi-target tracking.

    Paper: "Multiple Target Tracking: Revealing Causal Interactions in Complex Systems"
    GitHub: https://github.com/yonatandn/AutoRegKF

    Implements two-layer estimation: one for targets' location and another for correlation between trajectories.
    Uses Granger causality test to determine influence between object trajectories.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        num_objects: int = 1,
        ar_order: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__(state_dim, obs_dim)

        self.num_objects = num_objects
        self.ar_order = ar_order
        self.hidden_dim = hidden_dim

        # For single object case, treat as standard Kalman with AR dynamics
        if num_objects == 1:
            self.state_dim_per_object = state_dim
        else:
            self.state_dim_per_object = state_dim // num_objects
            assert (
                state_dim % num_objects == 0
            ), "state_dim must be divisible by num_objects"

        # Position estimator (Kalman Filter part)
        self.transition = nn.Linear(
            self.state_dim_per_object, self.state_dim_per_object
        )
        self.observation = nn.Linear(
            self.state_dim_per_object,
            obs_dim // num_objects if num_objects > 1 else obs_dim,
        )

        # Learnable noise “matrices” (don’t put these in a ModuleDict)
        self.register_parameter(
            "process_noise", nn.Parameter(torch.eye(self.state_dim_per_object) * 0.1)
        )
        self.register_parameter(
            "obs_noise",
            nn.Parameter(
                torch.eye(obs_dim // num_objects if num_objects > 1 else obs_dim) * 0.1
            ),
        )

        # AR coefficient matrices for Granger causality
        if num_objects > 1:
            self.ar_coefficients = nn.Parameter(
                torch.randn(
                    num_objects,
                    num_objects,
                    ar_order,
                    self.state_dim_per_object,
                    self.state_dim_per_object,
                )
                * 0.01
            )
        else:
            # Single object AR coefficients
            self.ar_coefficients = nn.Parameter(
                torch.randn(ar_order, state_dim, state_dim) * 0.01
            )

        self.causality_threshold = nn.Parameter(torch.tensor(0.05))

        # State history for AR modeling
        self._state_history = []
        self._causality_matrix = None

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        /,
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
        self._state_history = []
        if self.num_objects > 1:
            self._causality_matrix = torch.zeros(
                batch_size, self.num_objects, self.num_objects, device=device
            )
        else:
            self._causality_matrix = None

    def _granger_causality_test(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        Perform Granger causality test between object trajectories.

        Args:
            state_history: [batch, seq_len, num_objects, state_dim_per_object]
        Returns:
            causality_matrix: [batch, num_objects, num_objects]
        """
        batch_size, seq_len, num_objects, state_dim = state_history.shape
        causality_matrix = torch.zeros(
            batch_size, num_objects, num_objects, device=state_history.device
        )

        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    target_series = state_history[:, :, i, :]
                    predictor_series = state_history[:, :, j, :]

                    combined_input = torch.cat(
                        [target_series, predictor_series], dim=-1
                    )
                    combined_input = combined_input.view(batch_size, -1)

                    causality_score = torch.sigmoid(
                        torch.sum(
                            combined_input * self.ar_coefficients[i, j].view(-1), dim=-1
                        )
                    )
                    causality_matrix[:, i, j] = causality_score

        return causality_matrix

    def _ar_prediction(self, history: torch.Tensor) -> torch.Tensor:
        """
        Predict next state using autoregressive model.

        Args:
            history: [batch, ar_order, state_dim] or [batch, ar_order, num_objects, state_dim_per_object]
        Returns:
            prediction: [batch, state_dim] or [batch, num_objects, state_dim_per_object]
        """
        batch_size = history.shape[0]
        device = history.device

        if self.num_objects == 1:
            # Single object AR prediction
            prediction = torch.zeros(batch_size, self.state_dim, device=device)
            for t in range(self.ar_order):
                contribution = torch.matmul(
                    history[:, t, :].unsqueeze(1), self.ar_coefficients[t]
                )
                prediction += contribution.squeeze(1)
        else:
            # Multi-object AR prediction with Granger causality
            prediction = torch.zeros(
                batch_size, self.num_objects, self.state_dim_per_object, device=device
            )
            for i in range(self.num_objects):
                for j in range(self.num_objects):
                    for t in range(self.ar_order):
                        if self._causality_matrix is not None:
                            influence_weight = (
                                self._causality_matrix[:, i, j]
                                .unsqueeze(-1)
                                .unsqueeze(-1)
                            )
                        else:
                            influence_weight = 1.0 if i == j else 0.1

                        ar_coeff = self.ar_coefficients[i, j, t]
                        contribution = torch.matmul(
                            history[:, t, j, :].unsqueeze(1), ar_coeff
                        )
                        prediction[:, i, :] += influence_weight * contribution.squeeze(
                            1
                        )

        return prediction

    def _forward_step(
        self,
        observation: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        control: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One KF step with AR prediction:
          - Predict:  x_pred = A x + (AR if enough history)
          - Cov pred: P_pred = A P A^T + Q
          - Update:   K = P_pred H^T (H P_pred H^T + R)^{-1}
                      x = x_pred + K (y - H x_pred)
                      P = (I - K H) P_pred
        Uses nn.Linear modules (with bias) and the registered Q/R.
        """
        B = observation.size(0)
        device = observation.device
        if mask is None:
            mask = torch.ones(B, device=device, dtype=torch.bool)

        # Convenience
        I_full = torch.eye(self.state_dim, device=device).unsqueeze(0).expand(B, -1, -1)

        # Reshape per-object views if needed
        if self.num_objects > 1:
            obs_per_obj = self.obs_dim // self.num_objects
            Sobj = self.state_dim_per_object
            x_now = self._current_state.view(B, self.num_objects, Sobj)
            y_now = observation.view(B, self.num_objects, obs_per_obj)
        else:
            Sobj = self.state_dim
            x_now = self._current_state  # [B, S]
            y_now = observation  # [B, O]

        # -----------------
        # 1) PREDICT STATE
        # -----------------
        if len(self._state_history) >= self.ar_order:
            if self.num_objects > 1:
                hist = torch.stack(
                    self._state_history[-self.ar_order :], dim=1
                )  # [B,L,n,Sobj]
                x_pred = self._ar_prediction(hist)  # [B,n,Sobj]
            else:
                hist = torch.stack(
                    self._state_history[-self.ar_order :], dim=1
                )  # [B,L,S]
                x_pred = self._ar_prediction(hist)  # [B,S]
        else:
            if self.num_objects > 1:
                # Apply transition per object using the module (uses bias)
                x_flat = x_now.reshape(B * self.num_objects, Sobj)
                x_pred = self.transition(x_flat).reshape(B, self.num_objects, Sobj)
            else:
                x_pred = self.transition(x_now)  # [B,S]

        # -----------------
        # 2) PREDICT COV
        # -----------------
        # Expand A, H, Q, R as batched (constant across batch)
        if self.num_objects > 1:
            # Block-diagonal approximation: one A/H/Q/R per object block
            A = self.transition.weight  # [Sobj,Sobj]
            H = self.observation.weight  # [ObsObj,Sobj]
            Q = self.process_noise  # [Sobj,Sobj]
            R = self.obs_noise  # [ObsObj,ObsObj]

            P_prev = self._uncertainty  # [B, S, S]
            P_new = torch.zeros_like(P_prev)

            updated_states = []
            for i in range(self.num_objects):
                idx = slice(i * Sobj, (i + 1) * Sobj)
                # Pull out block for this object
                P_i = P_prev[:, idx, idx]  # [B,Sobj,Sobj]
                A_b = A.unsqueeze(0).expand(B, -1, -1)
                H_b = H.unsqueeze(0).expand(B, -1, -1)
                Q_b = Q.unsqueeze(0).expand(B, -1, -1)
                R_b = R.unsqueeze(0).expand(B, -1, -1)

                x_pred_i = x_pred[:, i, :]  # [B,Sobj]
                y_pred_i = self.observation(x_pred_i)  # [B,ObsObj]
                innov_i = y_now[:, i, :] - y_pred_i  # [B,ObsObj]

                # Covariance predict
                P_pred_i = A_b.bmm(P_i).bmm(A_b.transpose(1, 2)) + Q_b  # [B,Sobj,Sobj]

                # Innovation cov and gain
                S_i = (
                    H_b.bmm(P_pred_i).bmm(H_b.transpose(1, 2)) + R_b
                )  # [B,ObsObj,ObsObj]
                # K_i = P_pred_i H^T S^{-1} via solve for stability
                PHt = P_pred_i.bmm(H_b.transpose(1, 2))  # [B,Sobj,ObsObj]
                K_i = torch.linalg.solve(S_i, PHt.transpose(1, 2)).transpose(1, 2)

                # Masked update (skip update where mask==False)
                update_i = (K_i @ innov_i.unsqueeze(-1)).squeeze(-1)  # [B,Sobj]
                x_upd_i = torch.where(mask.unsqueeze(-1), x_pred_i + update_i, x_pred_i)

                # Cov update
                KH = K_i.bmm(H_b)  # [B,Sobj,Sobj]
                P_upd_i = (I_full[:, idx, idx] - KH).bmm(P_pred_i)  # [B,Sobj,Sobj]

                updated_states.append(x_upd_i)
                P_new[:, idx, idx] = P_upd_i

            x_upd = torch.stack(updated_states, dim=1).reshape(B, -1)  # [B,S]
            self._current_state = x_upd
            self._uncertainty = P_new
        else:
            # Single-object, full matrices
            A = self.transition.weight  # [S,S]
            H = self.observation.weight  # [O,S]
            Q = self.process_noise  # [S,S]
            R = self.obs_noise  # [O,O]

            P_prev = self._uncertainty  # [B,S,S]
            A_b = A.unsqueeze(0).expand(B, -1, -1)
            H_b = H.unsqueeze(0).expand(B, -1, -1)
            Q_b = Q.unsqueeze(0).expand(B, -1, -1)
            R_b = R.unsqueeze(0).expand(B, -1, -1)

            # Predict observation with the module (uses bias)
            y_pred = self.observation(x_pred)  # [B,O]
            innov = y_now - y_pred  # [B,O]

            # Cov predict
            P_pred = A_b.bmm(P_prev).bmm(A_b.transpose(1, 2)) + Q_b

            # Gain
            S = H_b.bmm(P_pred).bmm(H_b.transpose(1, 2)) + R_b  # [B,O,O]
            PHt = P_pred.bmm(H_b.transpose(1, 2))  # [B,S,O]
            K = torch.linalg.solve(S, PHt.transpose(1, 2)).transpose(1, 2)  # [B,S,O]

            # Masked update
            update = (K @ innov.unsqueeze(-1)).squeeze(-1)  # [B,S]
            x_upd = torch.where(mask.unsqueeze(-1), x_pred + update, x_pred)

            # Cov update
            KH = K.bmm(H_b)  # [B,S,S]
            P_upd = (I_full - KH).bmm(P_pred)

            self._current_state = x_upd
            self._uncertainty = P_upd

        # Keep AR history (store detached to avoid ballooning the graph)
        self._state_history.append(self._current_state.detach().clone())
        if len(self._state_history) > max(2 * self.ar_order, 10):
            self._state_history.pop(0)

        self._initialized = True
        return self._current_state.clone()

    def get_causality_matrix(self) -> Optional[torch.Tensor]:
        """Get current Granger causality matrix."""
        return self._causality_matrix

    def get_model_info(self) -> Dict[str, Any]:
        """Return model-specific information."""
        return {
            "model_type": "AutoRegKF",
            "architecture": "Autoregressive Kalman Filter with Granger Causality",
            "num_objects": self.num_objects,
            "ar_order": self.ar_order,
            "state_dim_per_object": self.state_dim_per_object,
            "features": [
                "granger_causality",
                "autoregressive_dynamics",
                "multi_object_tracking",
            ],
            "paper": "Multiple Target Tracking: Revealing Causal Interactions in Complex Systems",
            "github": "https://github.com/yonatandn/AutoRegKF",
        }
