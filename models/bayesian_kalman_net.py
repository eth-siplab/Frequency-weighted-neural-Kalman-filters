# bayesian_kalman_net.py

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_kalman_net import BaseKalmanNet
from .kalman_net import (
    LorenzTransition,
    MultiLayerPerceptron,
    QuadraticEmission,
    SinusoidalTransition,
    SphericalEmission,
)


class BayesianKalmanNet(BaseKalmanNet):
    """
    Bayesian KalmanNet using Monte Carlo Dropout for uncertainty quantification.

    Paper: "Uncertainty Quantification in Deep Learning Based Kalman Filters"
    ArXiv: https://arxiv.org/abs/2309.03058
    GitHub: https://github.com/yonatandn/Uncertainty-Quantification-in-Model-Based-DL

    Implements Bayesian deep learning with Monte Carlo dropout to provide both state
    estimates and associated uncertainty quantification.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        control_dim: int = 0,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        dropout_p: float = 0.1,
        num_mc_samples: int = 20,
        transition_model: str = "mlp",  # options: "mlp", "sinusoid", "lorenz", "identity"
        emission_model: str = "mlp",  # options: "mlp", "quadratic", "spherical", "identity"
    ):
        super().__init__(state_dim, obs_dim, control_dim)

        self.m = state_dim
        self.n = obs_dim
        self.k = control_dim

        # ---------------------------
        # Build transition f(x[,u])
        # ---------------------------
        t = transition_model.lower()
        if t == "mlp":
            # f takes [x,u] if control_dim>0 else just x
            f_in = self.m + (self.k if self.k and self.k > 0 else 0)
            self.f_in = f_in
            # two modest hidden layers; scale with dims but keep it sane
            f_h1 = max(32, math.sqrt(self.m * self.n))
            f_h2 = max(32, math.sqrt(self.m * self.n))
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
            h_h1 = max(32, math.sqrt(self.m * self.n))
            h_h2 = max(32, math.sqrt(self.m * self.n))
            self.h = MultiLayerPerceptron(self.m, h_h1, h_h2, output_dim=self.n)
        elif e == "quadratic":
            self.h = QuadraticEmission()
        elif e == "spherical":
            self.h = SphericalEmission()
        elif e == "identity":
            self.h = nn.Identity()
        else:
            raise ValueError(f"Unknown emission_model '{emission_model}'")

        self.num_mc_samples = num_mc_samples  # J in paper
        self.dropout_p = dropout_p

        if hidden_dim is None:
            hidden_dim = 10 * (state_dim**2 + obs_dim**2)

        # Input features: F2 and F4 (innovation and update differences)
        self.input_dim = obs_dim + state_dim

        # Network with dropout in FC layers for Bayesian inference
        # Reference: Paper Algorithm 1, Bayesian KalmanNet
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Dropout(dropout_p),  # Dropout for Bayesian inference
            nn.Tanh(),
        )

        # GRU with dropout for multiple layers
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0,
        )

        # Output layer with dropout
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_p), nn.Linear(hidden_dim, state_dim * obs_dim)
        )

        # Internal states
        self._gru_hidden = None
        self._kg_uncertainty = None  # Store Kalman gain uncertainty

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
        self._gru_hidden = None
        self._kg_uncertainty = torch.zeros(
            batch_size, self.state_dim, self.obs_dim, device=device
        )

    def _compute_features(
        self, obs: torch.Tensor, control: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute F2 and F4 features for Bayesian KalmanNet.

        F2: innovation difference Δy = y_t - ŷ_{t|t-1}
        F4: forward update difference Δx̂ = x̂_{t|t} - x̂_{t|t-1}
        """
        # Predict observation from current state
        obs_pred = self.h(self._current_state)

        # F2: Innovation difference
        innovation = obs - obs_pred

        # F4: State difference (simplified)
        state_pred = self.f(
            self._current_state, control if self.f_uses_control else None
        )
        state_diff = self._current_state - state_pred

        # Normalize features
        innovation = F.normalize(innovation, p=2, dim=1, eps=1e-12)
        state_diff = F.normalize(state_diff, p=2, dim=1, eps=1e-12)

        return torch.cat([innovation, state_diff], dim=-1)

    def _monte_carlo_kalman_gain(
        self, features: torch.Tensor, num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate Kalman gain using Monte Carlo sampling.

        Reference: Paper Algorithm 1, Bayesian KalmanNet
        "Sample {θ_j}_{j=1}^{J,o}: q(θ|φ);" - line 4

        Args:
            features: [batch, feature_dim] input features
            num_samples: number of MC samples (if None, uses self.num_mc_samples)

        Returns:
            kg_mean: [batch, state_dim, obs_dim] - mean Kalman gain
            kg_var: [batch, state_dim, obs_dim] - Kalman gain variance
        """
        if num_samples is None:
            num_samples = self.num_mc_samples

        # Enable dropout for MC sampling - Reference: Algorithm 1, line 5
        self.train()  # "for j = 1, 2, ..., J do"

        kg_samples = []

        # Monte Carlo sampling loop - Reference: Algorithm 1, lines 6-8
        for _ in range(num_samples):
            x = self.input_layer(features)
            x = x.unsqueeze(1)  # Add sequence dimension

            gru_out, hidden = self.gru(x, self._gru_hidden)
            kg_flat = self.output_layer(gru_out.squeeze(1))
            kg = kg_flat.view(-1, self.state_dim, self.obs_dim)
            kg_samples.append(kg)

        # Compute statistics across MC samples
        # Reference: Paper equations (22a) and (22b)
        kg_samples = torch.stack(
            kg_samples, dim=0
        )  # [num_samples, batch, state_dim, obs_dim]
        kg_mean = torch.mean(kg_samples, dim=0)  # x̂_t = (1/J) Σ x_t^{(j)}(y_t, θ_j)
        kg_var = torch.var(
            kg_samples, dim=0
        )  # Σ_t = (1/J) Σ (x_t^{(j)} - x̂_t)(x_t^{(j)} - x̂_t)^T

        return kg_mean, kg_var

    def _forward_step(
        self,
        observation: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        control: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process a single observation step with Bayesian uncertainty quantification.

        Args:
            observation: [batch, obs_dim]
            mask: [batch] - validity mask

        Returns:
            filtered_state: [batch, state_dim]
        """
        batch_size = observation.shape[0]
        device = observation.device

        if mask is None:
            mask = torch.ones(batch_size, device=device)

        # Predict step
        if self._initialized:
            self._current_state = self.f(
                self._current_state, control if self.f_uses_control else None
            )
        else:
            # Initialize with observation if possible
            if self.state_dim >= self.obs_dim:
                self._current_state[:, : self.obs_dim] = observation
            self._initialized = True

        # Compute features for Kalman Gain estimation
        features = self._compute_features(observation, control)

        # Estimate Kalman Gain with uncertainty using Monte Carlo sampling
        kalman_gain_mean, kalman_gain_var = self._monte_carlo_kalman_gain(features)

        # Store uncertainty for access
        self._kg_uncertainty = kalman_gain_var

        # Update step with mean Kalman gain
        obs_pred = self.h(self._current_state)
        innovation = observation - obs_pred

        # Apply mask
        masked_innovation = innovation * mask.unsqueeze(-1)

        # Kalman update: x_{t|t} = x_{t|t-1} + K_t * innovation
        update = torch.bmm(kalman_gain_mean, masked_innovation.unsqueeze(-1)).squeeze(
            -1
        )
        self._current_state = self._current_state + update

        # Update uncertainty estimate
        # Reference: Paper discussion on predictive uncertainty
        if self._uncertainty is not None:
            # Incorporate epistemic uncertainty from Kalman gain into state uncertainty
            H = (
                torch.eye(self.obs_dim, self.state_dim, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )

            # Mean update
            KH_mean = torch.bmm(kalman_gain_mean, H)
            I = (
                torch.eye(self.state_dim, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )

            # Basic uncertainty update (could be enhanced with full Bayesian treatment)
            self._uncertainty = torch.bmm(I - KH_mean, self._uncertainty)

            # Add epistemic uncertainty from Kalman gain variance
            # Build a batch of diagonal matrices from the per-dimension variance of K
            # kalman_gain_var: [B, state_dim, obs_dim] -> sum over obs_dim -> [B, state_dim]
            kg_uncertainty_diag = torch.sum(kalman_gain_var, dim=-1)  # [B, state_dim]
            kg_uncertainty_effect = torch.diag_embed(
                kg_uncertainty_diag
            )  # [B, state_dim, state_dim]
            # Add a small fraction to the state covariance (epistemic via K variance)
            self._uncertainty = self._uncertainty + 0.01 * kg_uncertainty_effect

        return self._current_state.clone()

    def forward(
        self,
        observations: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        controls: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ) -> torch.Tensor:
        """
        Extended forward pass that can return both epistemic and aleatoric uncertainty.

        Args:
            observations: [batch, seq_len, obs_dim]
            mask: [batch, seq_len] - optional validity mask
            initial_state: [batch, state_dim] - optional initial state
            controls: [batch, seq_len, control_dim] - optional control inputs
            return_uncertainty: whether to return uncertainty estimates

        Returns:
            states: [batch, seq_len, state_dim] - filtered state estimates
            uncertainties: [batch, seq_len, state_dim, state_dim] - if return_uncertainty=True
        """
        # Use base class implementation but also track Kalman gain uncertainties
        if return_uncertainty:
            batch_size, seq_len, _ = observations.shape
            device = observations.device

            # Initialize
            self.reset_state(batch_size, device)
            if initial_state is not None:
                self._current_state = initial_state

            if mask is None:
                mask = torch.ones(batch_size, seq_len, device=device)

            states = []
            uncertainties = []
            kg_uncertainties = []  # Track Kalman gain uncertainties

            for t in range(seq_len):
                obs_t = observations[:, t]
                mask_t = mask[:, t]

                state_t = self._forward_step(obs_t, mask=mask_t)
                states.append(state_t)

                if self._uncertainty is not None:
                    uncertainties.append(self._uncertainty.clone())

                if self._kg_uncertainty is not None:
                    kg_uncertainties.append(self._kg_uncertainty.clone())

            states = torch.stack(states, dim=1)
            uncertainties = torch.stack(uncertainties, dim=1) if uncertainties else None
            kg_uncertainties = (
                torch.stack(kg_uncertainties, dim=1) if kg_uncertainties else None
            )

            return states, {
                "state_uncertainty": uncertainties,
                "kalman_gain_uncertainty": kg_uncertainties,
            }
        else:
            return super().forward(
                observations,
                mask=mask,
                initial_state=initial_state,
                controls=controls,
                return_uncertainty=return_uncertainty,
            )

    def get_kalman_gain_uncertainty(self) -> torch.Tensor:
        """Get current Kalman gain uncertainty."""
        return self._kg_uncertainty

    def predict_with_uncertainty(
        self, observation: torch.Tensor, num_samples: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict state with full uncertainty quantification.

        Returns:
            state_mean: [batch, state_dim] - mean state estimate
            state_uncertainty: [batch, state_dim, state_dim] - predictive uncertainty
        """
        if num_samples is None:
            num_samples = self.num_mc_samples

        # Multiple forward passes with dropout
        self.train()
        state_samples = []

        for _ in range(num_samples):
            state_sample = self._forward_step(observation)
            state_samples.append(state_sample)

        # Compute statistics
        state_samples = torch.stack(
            state_samples, dim=0
        )  # [num_samples, batch, state_dim]
        state_mean = torch.mean(state_samples, dim=0)
        state_var = torch.var(state_samples, dim=0)

        # Convert variance to covariance matrix (simplified)
        state_mean.shape[0]
        state_uncertainty = torch.diag_embed(state_var)

        return state_mean, state_uncertainty

    def get_model_info(self) -> Dict[str, Any]:
        """Return model-specific information."""
        return {
            "model_type": "BayesianKalmanNet",
            "architecture": "KalmanNet with Monte Carlo Dropout for uncertainty quantification",
            "num_mc_samples": self.num_mc_samples,
            "dropout_p": self.dropout_p,
            "uncertainty_types": ["epistemic", "aleatoric", "kalman_gain_uncertainty"],
            "features": [
                "F2_innovation_diff",
                "F4_update_diff",
                "monte_carlo_sampling",
            ],
            "paper": "Uncertainty Quantification in Deep Learning Based Kalman Filters",
            "arxiv": "https://arxiv.org/abs/2309.03058",
            "github": "https://github.com/yonatandn/Uncertainty-Quantification-in-Model-Based-DL",
        }
