# recurrent_kalman_networks.py
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_kalman_net import BaseKalmanNet

TWO_PI = torch.tensor(2 * torch.pi)


class RKNTransitionCell(nn.Module):
    """
    RKN Transition Cell implementing factorized inference in high-dimensional space.

    Based on paper: "Recurrent Kalman Networks: Factorized Inference in
                     High-Dimensional Deep Feature Spaces"
    ArXiv: https://arxiv.org/abs/1905.07357

    Uses locally linear transition models with factorized covariances.
    """

    def __init__(
        self,
        latent_obs_dim: int,
        latent_state_dim: int,
        num_basis: int = 15,
        bandwidth: int = 3,
        trans_net_hidden_units: Optional[list] = None,
    ):
        super().__init__()

        self.latent_obs_dim = latent_obs_dim
        self.latent_state_dim = latent_state_dim
        self.num_basis = num_basis
        self.bandwidth = bandwidth

        if trans_net_hidden_units is None:
            trans_net_hidden_units = [60]
        self.trans_net_hidden_units = trans_net_hidden_units

        # Transition network to compute basis weights
        # Maps from current state to basis combination weights
        layers = []
        input_dim = latent_state_dim

        for hidden_dim in trans_net_hidden_units:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ELU()])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_basis))
        layers.append(nn.Softmax(dim=-1))  # Normalize basis weights

        self.transition_net = nn.Sequential(*layers)

        # Learnable basis matrices for locally linear transitions
        # Each basis matrix represents a local linear dynamic
        self.basis_matrices = nn.Parameter(
            torch.randn(num_basis, latent_state_dim, latent_state_dim) * 0.01
        )

        # Factorized covariance parameters
        # Process noise factors (low-rank + diagonal structure)
        self.process_noise_factors = nn.Parameter(
            torch.randn(latent_state_dim, bandwidth) * 0.1
        )
        self.process_noise_diag = nn.Parameter(torch.ones(latent_state_dim) * 0.1)

        # Observation noise parameters
        self.obs_noise_factors = nn.Parameter(
            torch.randn(latent_obs_dim, bandwidth) * 0.1
        )
        self.obs_noise_diag = nn.Parameter(torch.ones(latent_obs_dim) * 0.1)

        # Observation model (simple linear projection)
        self.obs_model = nn.Linear(latent_state_dim, latent_obs_dim)

    def predict_step(
        self, state_mean: torch.Tensor, state_cov: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction step using locally linear dynamics.

        Args:
            state_mean: [batch, latent_state_dim]
            state_cov: [batch, latent_state_dim, latent_state_dim]

        Returns:
            pred_mean: [batch, latent_state_dim]
            pred_cov: [batch, latent_state_dim, latent_state_dim]
        """
        batch_size = state_mean.shape[0]
        state_mean.device

        # Compute basis weights using transition network
        basis_weights = self.transition_net(state_mean)  # [batch, num_basis]

        # Compute effective transition matrix as weighted combination of basis matrices
        # F_eff = Σ w_i * A_i where w_i are basis weights, A_i are basis matrices
        transition_matrix = torch.einsum(
            "bn,nij->bij", basis_weights, self.basis_matrices
        )  # [batch, latent_state_dim, latent_state_dim]

        # Predict state mean: x_{t+1|t} = F_eff * x_{t|t}
        pred_mean = torch.bmm(transition_matrix, state_mean.unsqueeze(-1)).squeeze(-1)

        # Predict state covariance: P_{t+1|t} = F_eff * P_{t|t} * F_eff^T + Q
        FP = torch.bmm(transition_matrix, state_cov)
        FPFT = torch.bmm(FP, transition_matrix.transpose(-2, -1))

        # Add factorized process noise: Q = U U^T + D
        # where U is low-rank factor, D is diagonal
        process_noise_lowrank = torch.mm(
            self.process_noise_factors, self.process_noise_factors.t()
        )
        process_noise_diag = torch.diag(torch.exp(self.process_noise_diag))
        process_noise = process_noise_lowrank + process_noise_diag

        pred_cov = FPFT + process_noise.unsqueeze(0).expand(batch_size, -1, -1)

        return pred_mean, pred_cov

    def update_step(
        self, pred_mean: torch.Tensor, pred_cov: torch.Tensor, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update step using factorized Kalman gain computation.

        Args:
            pred_mean: [batch, latent_state_dim]
            pred_cov: [batch, latent_state_dim, latent_state_dim]
            obs: [batch, latent_obs_dim]

        Returns:
            post_mean: [batch, latent_state_dim]
            post_cov: [batch, latent_state_dim, latent_state_dim]
            log_likelihood: [batch]
        """
        batch_size = pred_mean.shape[0]
        device = pred_mean.device

        # Observation prediction: y_{t|t-1} = H * x_{t|t-1}
        obs_pred = self.obs_model(pred_mean)  # [batch, latent_obs_dim]

        # Innovation: v_t = y_t - y_{t|t-1}
        innovation = obs - obs_pred  # [batch, latent_obs_dim]

        # Observation matrix H (from linear layer)
        H = self.obs_model.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Predicted observation covariance: S = H * P_{t|t-1} * H^T + R
        HP = torch.bmm(H, pred_cov)
        HPH = torch.bmm(HP, H.transpose(-2, -1))

        # Factorized observation noise: R = V V^T + D_obs
        obs_noise_lowrank = torch.mm(self.obs_noise_factors, self.obs_noise_factors.t())
        obs_noise_diag = torch.diag(torch.exp(self.obs_noise_diag))
        obs_noise = obs_noise_lowrank + obs_noise_diag

        innovation_cov = HPH + obs_noise.unsqueeze(0).expand(batch_size, -1, -1)

        # Kalman gain: K = P_{t|t-1} * H^T * S^{-1}
        PH = torch.bmm(pred_cov, H.transpose(-2, -1))

        # Stable inverse using Cholesky decomposition
        try:
            L = torch.linalg.cholesky(innovation_cov)
            inv_innovation_cov = torch.cholesky_inverse(L)
        except:
            # Fallback to pseudo-inverse
            inv_innovation_cov = torch.pinverse(innovation_cov)

        kalman_gain = torch.bmm(PH, inv_innovation_cov)

        # Posterior state mean: x_{t|t} = x_{t|t-1} + K * v_t
        post_mean = pred_mean + torch.bmm(
            kalman_gain, innovation.unsqueeze(-1)
        ).squeeze(-1)

        # Posterior covariance: P_{t|t} = (I - K * H) * P_{t|t-1}
        I = torch.eye(self.latent_state_dim, device=device).unsqueeze(0)
        I = I.expand(batch_size, -1, -1)
        KH = torch.bmm(kalman_gain, H)
        post_cov = torch.bmm(I - KH, pred_cov)

        # Log-likelihood computation
        try:
            L = torch.linalg.cholesky(innovation_cov)
            log_det = 2 * torch.sum(
                torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1
            )
        except:
            sign, log_det = torch.slogdet(innovation_cov)

        # Quadratic form: v^T S^{-1} v
        quad_form = torch.bmm(
            innovation.unsqueeze(1),
            torch.bmm(inv_innovation_cov, innovation.unsqueeze(-1)),
        ).squeeze()

        # Gaussian log-likelihood
        log_likelihood = -0.5 * (
            log_det + quad_form + self.latent_obs_dim * torch.log(TWO_PI)
        )

        return post_mean, post_cov, log_likelihood


class RecurrentKalmanNetwork(BaseKalmanNet):
    """
    Recurrent Kalman Network for high-dimensional state estimation.

    Paper: "Recurrent Kalman Networks: Factorized Inference in
            High-Dimensional Deep Feature Spaces"
    ArXiv: https://arxiv.org/abs/1905.07357
    GitHub: https://github.com/LCAS/RKN

    Key features:
    - Encoder-decoder architecture with RKN transition cell
    - Factorized covariances for computational efficiency
    - Locally linear dynamics in learned latent space
    - Handles high-dimensional observations (e.g., images)
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        latent_state_dim: int = 15,
        latent_obs_dim: int = 10,
        num_basis: int = 15,
        bandwidth: int = 3,
        encoder_hidden_units: Optional[List[int]] = None,
        decoder_hidden_units: Optional[List[int]] = None,
        trans_net_hidden_units: Optional[List[int]] = None,
    ):
        # Note: state_dim and obs_dim are the original dimensions
        # latent_state_dim and latent_obs_dim are the learned latent dimensions
        super().__init__(state_dim, obs_dim)

        self.original_state_dim = state_dim
        self.original_obs_dim = obs_dim
        self.latent_state_dim = latent_state_dim
        self.latent_obs_dim = latent_obs_dim

        # Default hidden unit configurations
        if encoder_hidden_units is None:
            encoder_hidden_units = [128, 64]
        if decoder_hidden_units is None:
            decoder_hidden_units = [64, 128]
        if trans_net_hidden_units is None:
            trans_net_hidden_units = [60]

        # =============================================
        # Encoder: obs_dim -> latent_obs_dim
        # =============================================
        encoder_layers = []
        input_dim = obs_dim

        for hidden_dim in encoder_hidden_units:
            encoder_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ELU()])
            input_dim = hidden_dim

        # Final layer outputs both mean and log-variance for latent observation
        encoder_layers.append(nn.Linear(input_dim, 2 * latent_obs_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # =============================================
        # RKN Transition Cell
        # =============================================
        self.rkn_cell = RKNTransitionCell(
            latent_obs_dim=latent_obs_dim,
            latent_state_dim=latent_state_dim,
            num_basis=num_basis,
            bandwidth=bandwidth,
            trans_net_hidden_units=trans_net_hidden_units,
        )

        # =============================================
        # Decoder: latent_state_dim -> state_dim
        # =============================================
        decoder_layers = []
        input_dim = latent_state_dim

        for hidden_dim in decoder_hidden_units:
            decoder_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ELU()])
            input_dim = hidden_dim

        # Output layer for state reconstruction
        decoder_layers.append(nn.Linear(input_dim, state_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Initial latent state distribution parameters
        self.initial_mean = nn.Parameter(torch.zeros(latent_state_dim))
        self.initial_cov_factor = nn.Parameter(torch.eye(latent_state_dim) * 0.1)

        # Current latent state (for sequential processing)
        self._latent_state_mean = None
        self._latent_state_cov = None

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        initial_state: Optional[torch.Tensor] = None,
        initial_covariance: Optional[torch.Tensor] = None,
    ):
        """Reset internal latent state."""
        super().reset_state(
            batch_size,
            device,
            initial_state=initial_state,
            initial_covariance=initial_covariance,
        )

        # Initialize latent state distribution
        self._latent_state_mean = (
            self.initial_mean.unsqueeze(0).expand(batch_size, -1).clone()
        )

        initial_cov = torch.mm(self.initial_cov_factor, self.initial_cov_factor.t())
        self._latent_state_cov = (
            initial_cov.unsqueeze(0).expand(batch_size, -1, -1).clone()
        )

    def encode_observation(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode high-dimensional observation to latent space.

        Args:
            obs: [batch, obs_dim]

        Returns:
            latent_obs_mean: [batch, latent_obs_dim]
            latent_obs_var: [batch, latent_obs_dim]
        """
        encoded = self.encoder(obs)  # [batch, 2 * latent_obs_dim]

        # Split into mean and log-variance
        mean, log_var = torch.chunk(encoded, 2, dim=-1)
        var = torch.exp(log_var).clamp(min=1e-6)  # Ensure positive variance

        return mean, var

    def decode_state(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state to original state space.

        Args:
            latent_state: [batch, latent_state_dim]

        Returns:
            decoded_state: [batch, state_dim]
        """
        return self.decoder(latent_state)

    def _forward_step(
        self, observation: torch.Tensor, mask: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """
        Process a single observation through the RKN.

        Args:
            observation: [batch, obs_dim]
            mask: [batch] - validity mask

        Returns:
            decoded_state: [batch, state_dim]
        """
        batch_size = observation.shape[0]
        device = observation.device

        if mask is None:
            mask = torch.ones(batch_size, device=device)

        # Encode observation to latent space
        latent_obs_mean, latent_obs_var = self.encode_observation(observation)

        # Sample latent observation (reparameterization trick for training)
        if self.training:
            eps = torch.randn_like(latent_obs_mean)
            latent_obs = latent_obs_mean + torch.sqrt(latent_obs_var) * eps
        else:
            latent_obs = latent_obs_mean

        # Prediction step in latent space
        pred_mean, pred_cov = self.rkn_cell.predict_step(
            self._latent_state_mean, self._latent_state_cov
        )

        # Update step in latent space
        post_mean, post_cov, log_likelihood = self.rkn_cell.update_step(
            pred_mean, pred_cov, latent_obs
        )

        # Apply mask (skip update if observation is invalid)
        valid_mask = mask.unsqueeze(-1)
        self._latent_state_mean = torch.where(valid_mask, post_mean, pred_mean)

        # For covariance, we need to expand mask to match tensor dimensions
        cov_mask = mask.unsqueeze(-1).unsqueeze(-1)
        self._latent_state_cov = torch.where(cov_mask, post_cov, pred_cov)

        # Decode latent state to original space
        decoded_state = self.decode_state(self._latent_state_mean)

        # Update base class state (for compatibility)
        self._current_state = decoded_state.clone()

        return decoded_state

    def get_latent_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current latent state distribution."""
        return self._latent_state_mean, self._latent_state_cov

    def forward_with_uncertainty(
        self,
        observations: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning both state estimates and uncertainty.

        Args:
            observations: [batch, seq_len, obs_dim]
            mask: [batch, seq_len] - optional validity mask
            return_latent: whether to return latent space representations

        Returns:
            Dictionary containing:
            - states: [batch, seq_len, state_dim] - decoded state estimates
            - uncertainties: [batch, seq_len, latent_state_dim, latent_state_dim] - latent covariances
            - log_likelihoods: [batch, seq_len] - observation log-likelihoods
            - latent_states: [batch, seq_len, latent_state_dim] - if return_latent=True
        """
        batch_size, seq_len, obs_dim = observations.shape
        device = observations.device

        # Initialize
        self.reset_state(batch_size, device)

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=device)

        # Storage for outputs
        states = []
        uncertainties = []
        log_likelihoods = []
        latent_states = []

        for t in range(seq_len):
            obs_t = observations[:, t]
            mask_t = mask[:, t]

            # Process observation
            state_t = self._forward_step(obs_t, mask=mask_t)

            # Store outputs
            states.append(state_t)
            uncertainties.append(self._latent_state_cov.clone())

            if return_latent:
                latent_states.append(self._latent_state_mean.clone())

            # Compute log-likelihood for this step
            latent_obs_mean, latent_obs_var = self.encode_observation(obs_t)
            if self.training:
                eps = torch.randn_like(latent_obs_mean)
                latent_obs = latent_obs_mean + torch.sqrt(latent_obs_var) * eps
            else:
                latent_obs = latent_obs_mean

            # Get prediction from previous step for likelihood computation
            if t > 0:
                pred_mean, pred_cov = self.rkn_cell.predict_step(
                    prev_latent_mean, prev_latent_cov
                )
                _, _, ll = self.rkn_cell.update_step(pred_mean, pred_cov, latent_obs)
                log_likelihoods.append(ll)
            else:
                # For first timestep, use zero log-likelihood
                log_likelihoods.append(torch.zeros(batch_size, device=device))

            # Store for next iteration
            self._latent_state_mean.clone()
            self._latent_state_cov.clone()

        # Stack results
        result = {
            "states": torch.stack(states, dim=1),
            "uncertainties": torch.stack(uncertainties, dim=1),
            "log_likelihoods": torch.stack(log_likelihoods, dim=1),
        }

        if return_latent:
            result["latent_states"] = torch.stack(latent_states, dim=1)

        return result

    def compute_loss(
        self,
        observations: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute RKN loss combining reconstruction and likelihood terms.

        Args:
            observations: [batch, seq_len, obs_dim]
            targets: [batch, seq_len, state_dim]
            mask: [batch, seq_len] - optional validity mask

        Returns:
            Dictionary of losses
        """
        forward_result = self.forward_with_uncertainty(observations, mask)
        predicted_states = forward_result["states"]
        log_likelihoods = forward_result["log_likelihoods"]

        if mask is not None:
            # Apply mask to losses
            mask_expanded = mask.unsqueeze(-1)
            reconstruction_loss = (
                F.mse_loss(
                    predicted_states * mask_expanded,
                    targets * mask_expanded,
                    reduction="sum",
                )
                / mask.sum()
            )

            likelihood_loss = -(log_likelihoods * mask).sum() / mask.sum()
        else:
            reconstruction_loss = F.mse_loss(predicted_states, targets)
            likelihood_loss = -log_likelihoods.mean()

        total_loss = reconstruction_loss + 0.1 * likelihood_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "likelihood_loss": likelihood_loss,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Return model-specific information."""
        return {
            "model_type": "RecurrentKalmanNetwork",
            "architecture": "Encoder-RKNTransitionCell-Decoder with factorized inference",
            "original_dims": {
                "state_dim": self.original_state_dim,
                "obs_dim": self.original_obs_dim,
            },
            "latent_dims": {
                "latent_state_dim": self.latent_state_dim,
                "latent_obs_dim": self.latent_obs_dim,
            },
            "key_features": [
                "Factorized covariances",
                "Locally linear dynamics",
                "High-dimensional observation handling",
                "End-to-end learning",
            ],
            "transition_model": {
                "type": "Locally linear with basis matrices",
                "num_basis": self.rkn_cell.num_basis,
                "bandwidth": self.rkn_cell.bandwidth,
            },
            "paper": "Recurrent Kalman Networks: Factorized Inference in High-Dimensional Deep Feature Spaces",
            "arxiv": "https://arxiv.org/abs/1905.07357",
            "github": "https://github.com/LCAS/RKN",
            "advantages": [
                "Handles high-dimensional observations (images)",
                "Computationally efficient factorized operations",
                "Principled uncertainty quantification",
                "Avoids matrix inversions through factorization",
            ],
        }
