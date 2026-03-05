# recursive_kalman_net.py

from typing import Any, Dict, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_kalman_net import BaseKalmanNet
from .kalman_net import MultiLayerPerceptron


class RecursiveKalmanNet(BaseKalmanNet):
    """
    Recursive KalmanNet using Joseph's formula for consistent error covariance estimation.

    Paper: "Recursive KalmanNet: Deep Learning-Augmented Kalman Filtering for State
           Estimation with Consistent Uncertainty Quantification"
    ArXiv: https://arxiv.org/abs/2506.11639v1
    """

    def __init__(
        self, state_dim: int, obs_dim: int, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the recursive KalmanNet model.

        Args:
            state_dim (int): Dimensionality of the state space.
            obs_dim (int): Dimensionality of the observation space.
            config (Optional[Dict[str, Any]]): Configuration dictionary for the model.
        """
        super().__init__(state_dim, obs_dim)
        self.tbptt_interval = 14  # detach every 14 steps (~sqrt(200))

        default_config = {
            "nb_layer_FC1": 2,
            "FC1_mult": 2,
            "nb_layer_GRU": 1,
            "hidden_size_mult": 2,
            "nb_layer_FC2": 2,
            "FC2_mult": 2,
            "weight_factor": 0.1,
            # Stabilization knobs
            "gain_bound": 1.0,  # bound for tanh on Kalman gain entries
            "chol_diag_eps": 1e-6,  # epsilon added to diag after softplus
            "R_jitter": 1e-6,  # jitter added to R
        }

        if config is None:
            config = default_config
        else:
            # Update default with provided config
            default_config.update(config)
            config = default_config

        # Architecture parameters from paper
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.output_size = state_dim * obs_dim  # Kalman gain output size

        nb_layer_FC1 = config["nb_layer_FC1"]
        FC1_mult = config["FC1_mult"]
        self.nbr_GRU = config["nb_layer_GRU"]
        self.hidden_size_mult = config["hidden_size_mult"]
        nb_layer_FC2 = config["nb_layer_FC2"]
        FC2_mult = config["FC2_mult"]
        weight_factor = config["weight_factor"]

        self.gain_bound = float(config.get("gain_bound", 1.0))
        self._chol_diag_eps = float(config.get("chol_diag_eps", 1e-6))
        self._R_jitter = float(config.get("R_jitter", 1e-6))

        # Learnable emission model h: state -> observation (handles nonlinear mappings)
        h_h1 = max(32, int(math.sqrt(state_dim * obs_dim)) * 4)
        h_h2 = max(32, int(math.sqrt(state_dim * obs_dim)) * 4)
        self.h = MultiLayerPerceptron(state_dim, h_h1, h_h2, output_dim=obs_dim)

        # Input features: innovation, state correction, jacobians, measurement difference
        # As per paper: F1-F4 features for RKN
        self.input_size = (
            2 * self.obs_dim + self.state_dim * self.obs_dim + self.state_dim
        )
        self.hidden_size = self.output_size * self.hidden_size_mult

        # =====================================================
        # First RNN: Kalman Gain Estimation (Θ1 in paper)
        # =====================================================

        # FC layers before GRU for gain estimation
        gain_fc1_layers = []
        for i in range(nb_layer_FC1):
            in_features = self.input_size if i == 0 else self.input_size * FC1_mult
            out_features = self.input_size * FC1_mult
            gain_fc1_layers.append(nn.Linear(in_features, out_features))
            gain_fc1_layers.append(nn.ReLU())
        self.gain_fc1 = nn.Sequential(*gain_fc1_layers)

        # GRU for gain estimation
        self.gain_gru = nn.GRU(
            self.input_size * FC1_mult, self.hidden_size, self.nbr_GRU, batch_first=True
        )

        # FC layers after GRU for gain estimation
        gain_fc2_layers = []
        for i in range(nb_layer_FC2):
            if i == 0:
                in_features = self.hidden_size
            else:
                in_features = self.hidden_size * FC2_mult

            if i == nb_layer_FC2 - 1:
                out_features = self.output_size  # Final output: Kalman gain
            else:
                out_features = self.hidden_size * FC2_mult

            gain_fc2_layers.append(nn.Linear(in_features, out_features))
            if i < nb_layer_FC2 - 1:  # No activation on final layer
                gain_fc2_layers.append(nn.ReLU())
        self.gain_fc2 = nn.Sequential(*gain_fc2_layers)

        # =====================================================
        # Second RNN: Cholesky Factor Estimation (Θ2 in paper)
        # =====================================================

        # Input for Cholesky factor network: includes Kalman gain from first network
        chol_input_size = self.input_size + self.output_size

        # FC layers before GRU for Cholesky factor
        chol_fc1_layers = []
        for i in range(nb_layer_FC1):
            in_features = chol_input_size if i == 0 else chol_input_size * FC1_mult
            out_features = chol_input_size * FC1_mult
            chol_fc1_layers.append(nn.Linear(in_features, out_features))
            chol_fc1_layers.append(nn.ReLU())
        self.chol_fc1 = nn.Sequential(*chol_fc1_layers)

        # GRU for Cholesky factor estimation
        chol_hidden_size = (self.obs_dim * (self.obs_dim + 1)) // 2  # measurement space
        self.chol_gru = nn.GRU(
            chol_input_size * FC1_mult,
            chol_hidden_size * self.hidden_size_mult,
            self.nbr_GRU,
            batch_first=True,
        )

        # FC layers after GRU for Cholesky factor
        chol_fc2_layers = []
        for i in range(nb_layer_FC2):
            if i == 0:
                in_features = chol_hidden_size * self.hidden_size_mult
            else:
                in_features = chol_hidden_size * FC2_mult

            if i == nb_layer_FC2 - 1:
                out_features = chol_hidden_size  # Cholesky factor elements
            else:
                out_features = chol_hidden_size * FC2_mult

            chol_fc2_layers.append(nn.Linear(in_features, out_features))
            if i < nb_layer_FC2 - 1:
                chol_fc2_layers.append(nn.ReLU())
        self.chol_fc2 = nn.Sequential(*chol_fc2_layers)

        # Hidden states
        self.gain_hidden = None
        self.chol_hidden = None

        # Previous states for feature computation
        self._previous_state = None
        self._previous_obs = None

        # Weight factor
        self.weight_factor = weight_factor

        # Internal state
        self._initialized = False
        self._current_state: Optional[torch.Tensor] = None
        self._uncertainty: Optional[torch.Tensor] = None

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _compute_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute the input features for both RNNs.

        Features include:
        F1: Innovation (ỹt) = y_t − ŷ_t
        F2: State correction Δx_t = x_t − x_{t-1}
        F3: Jacobian of observation model H_t (flattened)
        F4: Measurement difference Δy_t = y_t − y_{t-1}
        """
        batch_size = obs.shape[0]
        device = obs.device

        # F1: Innovation (residual between observation and prediction)
        if self._initialized:
            # Predict observation from current state using learned emission model
            obs_pred = self.h(self._current_state)
            innovation = obs - obs_pred  # ỹt
        else:
            innovation = torch.zeros(batch_size, self.obs_dim, device=device)

        # F2: Previous state correction (simplified as state difference)
        if self._initialized:
            state_correction = self._current_state - self._previous_state
        else:
            state_correction = torch.zeros(batch_size, self.state_dim, device=device)

        # F3: Jacobian of observation model (computed from learned emission model)
        # Enable grad locally so this works even under torch.no_grad() (e.g. evaluate)
        with torch.enable_grad():
            x_for_jac = self._current_state.detach().requires_grad_(True)
            y_for_jac = self.h(x_for_jac)
            jac = torch.zeros(
                batch_size, self.obs_dim, self.state_dim, device=device
            )
            for i in range(self.obs_dim):
                grad_outputs = torch.zeros_like(y_for_jac)
                grad_outputs[:, i] = 1.0
                (g,) = torch.autograd.grad(
                    y_for_jac, x_for_jac, grad_outputs=grad_outputs, retain_graph=True
                )
                jac[:, i, :] = g
        jacobian = jac.reshape(batch_size, -1)

        # F4: Measurement difference
        if self._initialized:
            measurement_diff = obs - self._previous_obs
        else:
            measurement_diff = torch.zeros(batch_size, self.obs_dim, device=device)

        # Combine all features
        features = torch.cat(
            [
                innovation,  # F1: obs_dim
                measurement_diff,  # F4: obs_dim
                jacobian,  # F3: state_dim * obs_dim
                state_correction,  # F2: state_dim
            ],
            dim=-1,
        )

        return features

    def _cholesky_to_matrix(self, chol_factors: torch.Tensor) -> torch.Tensor:
        """Convert Cholesky factor vector to lower triangular matrix in observation space."""
        batch_size = chol_factors.shape[0]
        device = chol_factors.device
        dtype = chol_factors.dtype

        L = torch.zeros(
            batch_size, self.obs_dim, self.obs_dim, device=device, dtype=dtype
        )

        # Fill lower triangular part
        tril_indices = torch.tril_indices(self.obs_dim, self.obs_dim, device=device)
        L[:, tril_indices[0], tril_indices[1]] = chol_factors

        # Stabilize the diagonal: softplus + eps
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag.copy_(F.softplus(diag) + self._chol_diag_eps)

        return L

    def _joseph_covariance_update(
        self, P_pred: torch.Tensor, K: torch.Tensor, H: torch.Tensor, Rt: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Joseph's formula for covariance update as per paper.

        Joseph's formula: P_{t|t} = (I - K_t H_t) P_{t|t-1} (I - K_t H_t)^T + K_t R_t K_t^T
        where R_t is the corrected noise covariance term.
        """
        batch_size = P_pred.shape[0]
        device = P_pred.device

        I = (
            torch.eye(self.state_dim, device=device, dtype=P_pred.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # Compute (I - K_t H_t)
        KH = torch.bmm(K, H)
        A = I - KH

        # Joseph's formula implementation
        # First term: A P_{t|t-1} A^T
        AP = torch.bmm(A, P_pred)
        APAT = torch.bmm(AP, A.transpose(-2, -1))

        # Second term: K_t R_t K_t^T (corrected noise covariance)
        KR = torch.bmm(K, Rt)
        KRKT = torch.bmm(KR, K.transpose(-2, -1))

        # Complete Joseph's formula
        P_updated = APAT + KRKT

        return P_updated

    # -------------------------------------------------------------------------
    # BaseKalmanNet interface
    # -------------------------------------------------------------------------

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        /,
        initial_state: Optional[torch.Tensor] = None,
        initial_covariance: Optional[torch.Tensor] = None,
    ):
        """
        Reset internal state for a new sequence of observations.
        Initializes current_state and uncertainty (covariance).
        """
        super().reset_state(
            batch_size,
            device,
            initial_state=initial_state,
            initial_covariance=initial_covariance,
        )
        # Initialize hidden states for both RNNs
        self.gain_hidden = torch.zeros(
            self.nbr_GRU, batch_size, self.hidden_size, device=device
        )
        chol_hidden_size = (self.obs_dim * (self.obs_dim + 1)) // 2
        self.chol_hidden = torch.zeros(
            self.nbr_GRU,
            batch_size,
            chol_hidden_size * self.hidden_size_mult,
            device=device,
        )

    def _detach_state(self):
        """Detach internal + GRU hidden states for truncated BPTT."""
        super()._detach_state()
        if self.gain_hidden is not None:
            self.gain_hidden = self.gain_hidden.detach()
        if self.chol_hidden is not None:
            self.chol_hidden = self.chol_hidden.detach()
        if self._previous_state is not None:
            self._previous_state = self._previous_state.detach()

    def _forward_step(
        self,
        observation: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        control: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a single forward step using the recursive KalmanNet update.

        Args:
            observation (torch.Tensor): Current observation vector [batch, obs_dim].
            mask (Optional[torch.Tensor]): Optional mask for variable sequence lengths.
            control (Optional[torch.Tensor]): Optional control input vector [batch, control_dim].

        Returns:
            torch.Tensor: Updated state estimate [batch, state_dim].
        """
        device = observation.device
        batch_size = observation.size(0)

        # Initialize if first step
        if not self._initialized:
            self._previous_state = (
                self._current_state.clone()
                if self._current_state is not None
                else torch.zeros(batch_size, self.state_dim, device=device)
            )
            self._previous_obs = observation.clone()
            self._initialized = True

        # Keep copies for masking rollback if needed
        x_prev = self._current_state.clone()
        P_prev = self._uncertainty.clone() if self._uncertainty is not None else None

        # Compute input features
        features = self._compute_features(observation)

        # ===================================================
        # First RNN: Estimate Kalman Gain
        # ===================================================

        # Process features through gain estimation network
        x1 = self.gain_fc1(features)
        x1 = x1.unsqueeze(1)  # Add sequence dimension

        gru_out1, self.gain_hidden = self.gain_gru(x1, self.gain_hidden)
        kalman_gain_flat = self.gain_fc2(gru_out1.squeeze(1))

        # Reshape to Kalman gain matrix and bound it for stability
        K_raw = kalman_gain_flat.view(batch_size, self.state_dim, self.obs_dim)
        if self.gain_bound is not None and self.gain_bound > 0:
            kalman_gain = torch.tanh(K_raw) * self.gain_bound
        else:
            kalman_gain = K_raw

        # ===================================================
        # Second RNN: Estimate Cholesky Factor
        # ===================================================

        # Combine features with estimated Kalman gain
        chol_input = torch.cat([features, kalman_gain_flat], dim=-1)

        x2 = self.chol_fc1(chol_input)
        x2 = x2.unsqueeze(1)

        gru_out2, self.chol_hidden = self.chol_gru(x2, self.chol_hidden)
        chol_factors = self.chol_fc2(gru_out2.squeeze(1))

        # Convert to Cholesky matrix and then to measurement noise covariance Rt
        L = self._cholesky_to_matrix(chol_factors)
        Rt = torch.bmm(L, L.transpose(-2, -1))
        Rt = Rt + self._R_jitter * torch.eye(
            self.obs_dim, device=device, dtype=Rt.dtype
        ).unsqueeze(0)

        # ===================================================
        # Kalman Update with Joseph's Formula
        # ===================================================

        # Predicted observation using learned emission model
        y_pred = self.h(self._current_state)

        # Compute observation Jacobian H via autograd for Joseph's formula
        # Enable grad locally so this works even under torch.no_grad() (e.g. evaluate)
        with torch.enable_grad():
            x_for_jac = self._current_state.detach().requires_grad_(True)
            y_for_jac = self.h(x_for_jac)
            H = torch.zeros(
                batch_size, self.obs_dim, self.state_dim,
                device=device, dtype=observation.dtype,
            )
            for i in range(self.obs_dim):
                grad_outputs = torch.zeros_like(y_for_jac)
                grad_outputs[:, i] = 1.0
                (g,) = torch.autograd.grad(
                    y_for_jac, x_for_jac, grad_outputs=grad_outputs, retain_graph=True
                )
                H[:, i, :] = g

        # Innovation
        innovation = observation - y_pred

        # State update
        state_update = torch.bmm(kalman_gain, innovation.unsqueeze(-1)).squeeze(-1)
        x_new = self._current_state + state_update

        # Covariance update using Joseph's formula
        if self._uncertainty is not None:
            P_new = self._joseph_covariance_update(
                self._uncertainty, kalman_gain, H.detach(), Rt
            )
        else:
            P_new = None

        # Apply mask if provided (rollback to previous when mask == 0)
        if mask is not None:
            m = mask.view(-1, 1).to(x_new.dtype)
            x_new = m * x_new + (1 - m) * x_prev
            if P_new is not None:
                m2 = m.view(batch_size, 1, 1)
                P_new = m2 * P_new + (1 - m2) * P_prev

        # Commit
        self._current_state = x_new
        if P_new is not None:
            self._uncertainty = P_new

        # Store previous for next step
        self._previous_state = self._current_state.clone()
        self._previous_obs = observation.clone()

        return self._current_state

    def sequence_nll(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sequence negative log-likelihood (NLL) for given observations and
        corresponding predicted states. If uncertainties are provided, include
        covariance-based likelihood, otherwise use squared error.

        Args:
            observations (torch.Tensor): Observations [batch, seq_len, obs_dim]
            states (torch.Tensor): Predicted states [batch, seq_len, state_dim]
            uncertainties (Optional[torch.Tensor]): Covariances [batch, seq_len, state_dim, state_dim]

        Returns:
            torch.Tensor: Mean NLL over sequence.
        """
        batch, seq_len, obs_dim = observations.shape
        device = observations.device

        log_likelihood = torch.tensor(0.0, device=device)

        for t in range(seq_len):
            y_t = observations[:, t, :]
            x_t = states[:, t, :]

            # Predicted observation using learned emission model
            y_pred_t = self.h(x_t)

            # Error
            error_t = y_t - y_pred_t

            if uncertainties is not None:
                # Extract observation covariance (assume H = I for simplicity)
                cov_t = uncertainties[
                    :, t, :obs_dim, :obs_dim
                ]  # [batch, obs_dim, obs_dim]

                # Log determinant and quadratic form
                try:
                    # Use Cholesky for stability
                    L = torch.linalg.cholesky(cov_t)
                    log_det = 2.0 * torch.sum(
                        torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1
                    )
                    v = torch.cholesky_solve(error_t.unsqueeze(-1), L).squeeze(-1)
                    quad_form = torch.sum(error_t * v, dim=-1)
                except RuntimeError:
                    # Fallback to pseudo-inverse if singular
                    cov_inv = torch.pinverse(cov_t)
                    quad_form = torch.bmm(
                        error_t.unsqueeze(1), torch.bmm(cov_inv, error_t.unsqueeze(-1))
                    ).squeeze()

                # Gaussian log-likelihood: -0.5 * (log|Σ| + error^T Σ^{-1} error + k*log(2π))
                nll_t = 0.5 * (
                    log_det
                    + quad_form
                    + obs_dim * torch.log(2 * torch.tensor(torch.pi))
                )

            else:
                # Simplified case without covariance
                nll_t = 0.5 * torch.sum(error_t**2, dim=-1)

            log_likelihood = log_likelihood + nll_t.mean()

        return log_likelihood / seq_len

    def get_model_info(self) -> Dict[str, Any]:
        """Return model-specific information."""
        return {
            "model_type": "RecursiveKalmanNet",
            "architecture": "Two separate RNNs with Joseph's formula for covariance update",
            "networks": {
                "gain_network": "Kalman gain estimation (Θ1)",
                "cholesky_network": "Noise covariance Cholesky factor estimation (Θ2)",
            },
            "features": [
                "F1_innovation",
                "F2_state_correction",
                "F3_jacobian",
                "F4_measurement_diff",
            ],
            "covariance_method": "Joseph's formula for consistent error covariance",
            "loss_function": "Gaussian negative log-likelihood",
            "paper": "Recursive KalmanNet: Deep Learning-Augmented Kalman Filtering",
            "arxiv": "https://arxiv.org/abs/2506.11639v1",
            "key_contributions": [
                "Joseph's formula implementation",
                "Consistent error covariance quantification",
                "No prior noise covariance knowledge required",
            ],
        }
